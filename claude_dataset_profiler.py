"""
claude_dataset_profiler.py
Nodo ComfyUI — Dataset Profiler / Auditor (primer nodo del sistema de captioning post-shift).
Parte del repo comfyui-rafa-nodes — github.com/osuvense/comfyui-rafa-nodes

Analiza una MUESTRA representativa del dataset (submuestreo determinista + baja resolución)
y emite un Dataset Profile en Markdown: invariantes propuestos a enmascarar, vocabulario
canónico de escena y reporte de curación. Una sola llamada a la API por dataset.

El perfil es revisado/editado por el operador (Rafa) antes de alimentar al Captioner
(segundo nodo). La máquina propone; el operador decide.

Diseño: [REF]-nodo-captioning.md § 5.1, § 5.7, § 6 (revisión 3, 29 may 2026).
"""

import os
import io
import base64
import anthropic
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Resolución del submuestreo (lado largo). Los invariantes son macro-rasgos:
# no requieren resolución plena (§ 5.1).
SAMPLE_MAX_SIDE = 768

# Los 5 modos confirmados (D7). El Profiler los usa para saber qué clase de
# invariante buscar.
CAPTION_MODES = [
    "Identidad-persona",
    "Anatomía-sujeto-específico",
    "Concepto-acción-pose",
    "Estilo",
    "Custom",
]

# Niveles de effort de la API (output_config.effort). Fix #5 (3 jun 2026).
EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]

# Modos de thinking. Opus 4.8/4.7 solo admiten adaptive|disabled (manual da 400).
# 'adaptive' primero = default del combo (decisión producción 3 jun, igual que el Captioner).
# En Opus 4.8/4.7 temperature se ignora en ambos (deprecada). Fix #5.
THINKING_MODES = ["adaptive", "disabled"]

# ============================================================
# SYSTEM PROMPT (v1, 29 may 2026 — [REF]-nodo-captioning.md § 5.7)
# ============================================================

PROFILER_SYSTEM_PROMPT = """You are the Dataset Profiler / Auditor, the first of two chained stages in a
LoRA-training caption system for ComfyUI. You analyze a representative,
low-resolution SAMPLE of an image dataset and emit a single structured Dataset
Profile. You do NOT caption individual images — a separate Captioner stage does
that, consuming your profile.

A human operator reviews and edits your profile before it reaches the Captioner.
Your role is to PROPOSE BY OBSERVATION; the operator decides. Never present a
guess as a fact, and never finalize a keep/discard decision yourself.

## The paradigm you serve: masked captioning

In this pipeline, whatever the captions DESCRIBE is excluded from what the LoRA
learns as identity; the model treats described traits as general knowledge.
Whatever the captions LEAVE UNDESCRIBED is bound to the trigger word as an
indivisible block. Captions are a mask: what they cover is not learned; what
they leave visible is learned and anchored to the trigger.

Anchoring also operates by LEXICAL REPETITION: a descriptor repeated identically
across the whole dataset gets anchored to the model even when it is not the
trigger. Two consequences you must respect:
- Consistent phrasing is DESIRABLE for scene elements that should stay stable.
- Describing something consistently that should NOT be anchored will fix it anyway.

Your entire job is to support this: decide, by looking at the sample as a SET,
what is invariant across this dataset (-> mask, the trigger absorbs it) versus
what varies (-> describe, so the LoRA does not fix it), and to give the Captioner
a consistent vocabulary for the variable scene.

## Observe, do not remember

Classify strictly from the visual evidence in the sample. You do NOT know the
subject or character; you only read technical signals. Do not import a fixed,
generic invariant checklist (ethnicity, hair color, eye color) and tick it off —
detect THIS dataset's actual invariants dynamically, including subject-specific
ones a generic list would never contain (a particular baldness pattern, a
specific moustache, one subject's specific anatomy). Macro-features only: you are
working at low resolution, so reason about coarse, stable features (build,
baldness, facial hair, anatomy, overall styling), not fine detail (thin chains,
small accessories) — fine detail is the per-image Captioner's job.

## What counts as invariant depends on caption_mode

The mode tells you what kind of invariant to look for:

- Identidad-persona — the dataset teaches one person's identity. Invariants =
  that person's permanent appearance, constant across the sample: build/weight,
  baldness/hair, moustache/beard/facial hair, body hair, facial structure, skin
  tone, eye color, feet, and — when it is consistently the same person and
  genitals are visible — genitals. These are masked.

- Anatomía-sujeto-específico — the dataset teaches one specific subject's anatomy
  (a single person's body/genitals). Everything anatomically permanent to that
  one subject is invariant and masked (e.g. penis, belly, legs, pubis,
  testicles). What VARIES and must be described: state (erect / semi / flaccid),
  foreskin position (retracted / covering the glans), hand grip, camera angle,
  lighting, background. The Captioner pairs this mode with clinical anatomical
  description of the variable elements.
  CRITICAL — state is not identity. One subject appears across many states. Do
  NOT infer a "second subject" or a different identity from a change of arousal
  state, foreskin position (retracted vs covering the glans), or lighting. Those
  are variable STATES of the SAME subject, exactly like a smile vs a neutral mouth
  on one person — not identity signals. Reason about macro identity only (build,
  proportions, characteristic shape), never about transient state or light.

- Concepto-acción-pose — the dataset teaches a concept, action or pose across
  DIFFERENT subjects. Here anatomy is NOT invariant (it changes per subject) and
  must be described, so the LoRA learns the concept and not one instance. Look
  instead for what is genuinely constant — the concept/action/composition being
  taught — as the thing anchored to the trigger.

- Estilo — the invariant is the visual style itself, absorbed by the trigger.
  Subject content varies and is described. Do not propose subject features as
  invariants; note the consistent stylistic markers only as context.

- Custom — follow the invariant definition the operator supplies, and still
  produce all three outputs against it.

## Output 1 — Proposed invariants (populates invariant_traits)

The traits that are constant across the sample, written as short English
descriptors. This list directly populates the Captioner's invariant_traits field
(what it will NOT describe). Split into:

- Confident — observed consistently across the whole sample -> recommend masking.
- Borderline — present in most but not all images, or ambiguous whether it is
  identity or a variable accessory (e.g. glasses in ~9 of 12 images). SURFACE THE
  PRESENCE RATE; DO NOT SET THE THRESHOLD — the invariant-vs-variable cutoff is
  the operator's decision.

Also list, separately, the VARIABLE traits you observed (things that change
across the sample and therefore must be described). This shows the operator both
sides of the cut.

## Output 2 — Canonical scene vocabulary

For lexical consistency. Propose ONE canonical English phrasing per recurring
SCENE / CONTEXT element, to be reused verbatim across captions (e.g. daytime
exterior -> "bright natural sunlight casting shadows"; pick one and stick to it
rather than alternating with "harsh midday light" for the same thing). Group as:
settings/locations, lighting types, recurring clothing/props.

Rules:
- Canonical vocabulary is for SCENE AND CONTEXT ONLY, never for the masked
  invariants.
- It must cover only elements that should be described consistently — never
  assign canonical phrasing to something that should remain variable per image.

## Output 3 — Curation report

Audit the sample for problems. Three checks plus quality:

AI-origin audit — five distinctions. For each flagged image, state which category
and your confidence:
1. Generative AI (fabricated image) -> recommend DISCARD.
2. Local cleanup AI (object removal, Apple Intelligence Clean Up and similar)
   -> KEEP.
3. Video-capture motion blur (localized blur with coherent texture elsewhere)
   -> KEEP. Do NOT misclassify this as generative AI — it is a documented
   false-positive trap.
4. Extreme-angle distortion (ultra-low selfie, lens distortion) -> KEEP.
5. Beauty / smoothing filter (Instagram/Snapchat beauty mode) -> KEEP; the
   smoothed look will be learned but does not contaminate identity.

Video-frame families. Group images that look like frames of the same video or
burst (same outfit + same location + near-identical lighting, differing only
slightly). For each family, say whether poses are varied (keep all) or redundant
(recommend thinning).

Overlays / UI / watermarks. Flag social-media overlays (Stories UI, captions,
avatars, stickers), platform watermarks and timestamps. Recommend pre-processing
(crop/clean) or discard if intrusive and uncroppable.

Low resolution / quality. Flag images too low-res or degraded to caption
reliably. Judge conservatively — you are seeing a downscaled sample.

## Reasoning rules

- Judge invariant-vs-variable from CONSISTENCY ACROSS THE SAMPLED SET, not from
  any single image and not from prior knowledge.
- Report presence rates; do not set thresholds. Borderline cases go to the
  operator.
- Calibrate evidence and mark uncertainty. Separate what the sample lets you
  confirm from judgment calls and from genuinely ambiguous cases. Never state a
  guess as confirmed.
- Do not invent. If a trait is not clearly observable across the sample, do not
  list it.
- If the sample looks unrepresentative or too small to judge a given invariant
  confidently, say so.

## Do not

- Do not caption individual images or produce any training caption text.
- Do not apply a generic invariant checklist blindly; detect this dataset's real,
  possibly subject-specific invariants.
- Do not finalize keep/discard — recommend; the operator decides.
- Do not put masked invariants into the canonical vocabulary.
- Do not attribute causes you cannot see; report observable signals only.

## Image labels

Each image in the sample is preceded by a text line "Filename: <name>". Use those
exact filenames in the Curation Report. Never number the images by order and never
invent a name; only reference filenames that were actually provided.

## Output format

Emit exactly this structure, in English:

## DATASET PROFILE
caption_mode: <mode>
sample analyzed: <n> images

### INVARIANTS TO MASK (proposed -> invariant_traits)
Confident (constant across sample -> recommend masking):
- <english descriptor> — <presence note>
Borderline (needs operator decision):
- <english descriptor> — <presence rate + why borderline>

### VARIABLE TRAITS (the Captioner must describe these)
- <english descriptor> — <how it varies across the sample>

### CANONICAL SCENE VOCABULARY
Settings / locations:
- <element> -> "<canonical phrasing>"
Lighting:
- <element> -> "<canonical phrasing>"
Recurring clothing / props:
- <element> -> "<canonical phrasing>"

### CURATION REPORT
AI-origin audit:
- <filename> — <category 1-5> — <keep/discard> — <confidence>
Video-frame families:
- Family <id> (<filenames>): <varied -> keep all / redundant -> thin>
Overlays / UI / watermarks:
- <filename> — <type> — <pre-process / discard>
Low resolution / quality:
- <filename> — <note>

### NOTES FOR OPERATOR REVIEW
- <threshold calls, sample representativeness, anything uncertain>"""


# ============================================================
# NODO
# ============================================================

# --- Compat SDK anthropic >= 1.0.0 (PyPI 20/08/2026): `temperature` ELIMINADA ---
# La 1.0.0 borro temperature/top_p/top_k de messages.create(). Ya no es un 400 de
# la API sino un TypeError LOCAL de Python ("unexpected keyword argument
# 'temperature'"), que el fallback antiguo no reconocia y escupia como
# "Error inesperado". Se detecta por introspeccion de la firma una sola vez por
# proceso y se omite el kwarg de entrada.
_SDK_HAS_TEMPERATURE = None


def _sdk_accepts_temperature() -> bool:
    global _SDK_HAS_TEMPERATURE
    if _SDK_HAS_TEMPERATURE is None:
        try:
            import inspect
            from anthropic.resources.messages import Messages
            _SDK_HAS_TEMPERATURE = "temperature" in inspect.signature(Messages.create).parameters
        except Exception:
            _SDK_HAS_TEMPERATURE = True  # ante la duda se intenta; el retry lo cubre
    return bool(_SDK_HAS_TEMPERATURE)


class ClaudeDatasetProfiler:
    """
    Dataset Profiler / Auditor. Submuestrea el dataset y emite un Dataset Profile
    (invariantes propuestos + vocabulario canónico + reporte de curación) en una
    sola llamada a la API. El perfil se revisa/edita antes de pasar al Captioner.
    """

    CATEGORY = "rafa"
    OUTPUT_NODE = True
    FUNCTION = "profile_dataset"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dataset_profile", "log")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_folder": ("STRING", {
                    "default": "/workspace/datasets/MiPersonaje/raw",
                    "multiline": False,
                    "tooltip": "Ruta de la carpeta con las imágenes del dataset. Se muestrea de aquí."
                }),
                "caption_mode": (CAPTION_MODES, {
                    "tooltip": (
                        "Tipo de objetivo del dataset. Determina qué clase de invariante busca el "
                        "Profiler. Debe coincidir con el caption_mode del Captioner."
                    )
                }),
                "sample_size": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": (
                        "Número de imágenes a muestrear (stride uniforme determinista sobre la "
                        "carpeta ordenada). Los invariantes se ven en cualquier subconjunto; 12 suele bastar."
                    )
                }),
                "model": ("STRING", {
                    "default": "claude-opus-4-8",
                    "tooltip": (
                        "Model string de Anthropic. Default claude-opus-4-8 (observación superior en la "
                        "validación, §5.9). Editable — actualizar cuando salgan modelos nuevos."
                    )
                }),
                "temperature": ("FLOAT", {
                    "default": 0.20,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "0.20 estable (no subir de 0.40). OJO: Opus 4.8/4.7 IGNORAN temperature (deprecada); usa 'effort'. El nodo la omite solo si el modelo la rechaza."
                }),
            },
            "optional": {
                "effort": (EFFORT_LEVELS, {
                    "default": "high",
                    "tooltip": (
                        "output_config.effort de la API (low/medium/high/xhigh/max). high = default de la API. "
                        "Controla cuántos tokens gasta el modelo. Requiere SDK anthropic>=0.105.2."
                    )
                }),
                "thinking": (THINKING_MODES, {
                    "default": "adaptive",
                    "tooltip": (
                        "adaptive = el modelo decide cuánto razonar (DEFAULT de producción; mejor razonamiento estado-vs-identidad). "
                        "Sube max_tokens y la latencia/coste. disabled = sin extended thinking, más rápido y barato. "
                        "En Opus 4.8/4.7 temperature se ignora en ambos casos (deprecada)."
                    )
                }),
                "prompt_caching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cachea el system prompt (cache_control ephemeral). Surfacea cache_w/cache_r en el log."
                }),
                "output_profile_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Vacío → el perfil sale solo por el output 'dataset_profile'. "
                        "Relleno → además se escribe a este fichero (.md) como documentación del dataset."
                    )
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Vacío → usa la variable de entorno ANTHROPIC_API_KEY (recomendado)."
                }),
            }
        }

    # ----------------------------------------------------------

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Nodo con side-effect (puede escribir el perfil a disco) y análisis que se
        # relanza; ComfyUI lo cachearía por inputs. NaN -> re-ejecuta en cada Queue.
        return float("nan")

    def profile_dataset(
        self,
        image_folder,
        caption_mode,
        sample_size,
        model,
        temperature,
        effort="high",
        thinking="adaptive",
        prompt_caching=True,
        output_profile_path="",
        api_key="",
    ):
        logs = []
        self._temp_unsupported = False

        # ---- Validaciones ----
        if not os.path.isdir(image_folder):
            return ("", f"[ERROR] La carpeta no existe: {image_folder}")

        key = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return ("", "[ERROR] No se encontró ANTHROPIC_API_KEY (ni en el nodo ni como variable de entorno).")

        images = sorted([
            f for f in os.listdir(image_folder)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])
        if not images:
            return ("", f"[ERROR] No se encontraron imágenes en: {image_folder}")

        total = len(images)
        sample = self._select_sample(images, sample_size)
        logs.append(f"[PROFILER] muestra {len(sample)}/{total} imágenes (stride determinista) — modo {caption_mode}")

        # ---- Codificar la muestra a baja resolución ----
        # Cada imagen va precedida de un bloque de texto "Filename: <name>" para que el
        # modelo mapee imagen->nombre real (fix #1a, 3 jun 2026: antes numeraba por orden
        # de envío e inventaba nombres como "img_13").
        user_content = []
        for fname in sample:
            try:
                img_b64, media_type = self._image_to_base64_downscaled(
                    os.path.join(image_folder, fname), SAMPLE_MAX_SIDE
                )
            except Exception as e:
                logs.append(f"[WARN] {fname} — omitida del muestreo (fallo al cargar): {e}")
                continue
            user_content.append({"type": "text", "text": f"Filename: {fname}"})
            user_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": img_b64},
            })

        if not any(c.get("type") == "image" for c in user_content):
            return ("", "[ERROR] No se pudo cargar ninguna imagen de la muestra.\n" + "\n".join(logs))

        n_imgs = sum(1 for c in user_content if c.get("type") == "image")
        user_content.append({
            "type": "text",
            "text": (
                f"Here is a representative sample of {n_imgs} images from the dataset, each "
                f"preceded by its 'Filename:' line. caption_mode: {caption_mode}. Analyze the "
                f"sample as a SET and emit the Dataset Profile in the exact output format, in "
                f"English. In the Curation Report, reference images by those exact filenames "
                f"only; never number them or invent names. Do not caption individual images."
            ),
        })

        # ---- System prompt (cacheable) ----
        system_param = [{"type": "text", "text": PROFILER_SYSTEM_PROMPT}]
        if prompt_caching:
            system_param[0]["cache_control"] = {"type": "ephemeral"}

        # ---- Llamada API (una sola) ----
        # Fix #5: thinking 'adaptive' es INCOMPATIBLE con temperature (doc Anthropic) -> se
        # omite temperature; 'disabled' la respeta. Opus 4.8/4.7 solo admiten adaptive|disabled.
        # SDK anthropic>=0.105.2 acepta output_config/thinking nativos; si un SDK viejo los
        # rechaza, _create_with_fallback los reenvía vía extra_body.
        client = anthropic.Anthropic(api_key=key)
        api_kwargs = dict(
            model=model.strip(),
            max_tokens=(16000 if thinking == "adaptive" else 8000),
            system=system_param,
            messages=[{"role": "user", "content": user_content}],
            output_config={"effort": effort},
        )
        if thinking == "adaptive":
            api_kwargs["thinking"] = {"type": "adaptive"}
        else:
            api_kwargs["temperature"] = temperature
        try:
            response = self._create_with_fallback(client, api_kwargs, logs)
            profile = self._extract_text(response).strip()
            usage = response.usage
            tok_in = usage.input_tokens
            tok_out = usage.output_tokens
            cache_w = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_r = getattr(usage, "cache_read_input_tokens", 0) or 0
        except Exception as e:
            return ("", f"[ERROR] API: {e}\n" + "\n".join(logs))
        if not profile:
            logs.append("[WARN] respuesta sin bloque de texto (¿el modelo solo emitió thinking?).")

        logs.append(f"[PROFILER] in:{tok_in} out:{tok_out} cache_w:{cache_w} cache_r:{cache_r}")

        # ---- Guardar perfil a fichero (opcional) ----
        if output_profile_path.strip():
            try:
                path = output_profile_path.strip()
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(profile)
                logs.append(f"[PROFILER] perfil guardado en {path}")
            except Exception as e:
                logs.append(f"[WARN] perfil generado pero no guardado: {e}")

        return (profile, "\n".join(logs))

    # ----------------------------------------------------------

    def _select_sample(self, images, sample_size):
        """Submuestreo por stride uniforme determinista (§ 6, revisión 3).
        Cambiar a aleatorio = 1 línea: return random.sample(images, k)."""
        n = len(images)
        k = max(1, min(int(sample_size), n))
        if k >= n:
            return list(images)
        stride = n / k
        return [images[int(i * stride)] for i in range(k)]

    # ----------------------------------------------------------

    def _create_with_fallback(self, client, api_kwargs, logs):
        """Llama a messages.create con dos salvaguardas:
        - Si el SDK no acepta output_config/thinking como kwargs nativos (versión
          < 0.105.2), los reenvía vía extra_body.
        - Si el modelo ha deprecado 'temperature' (Opus 4.8/4.7 devuelven 400
          '`temperature` is deprecated for this model.'), reintenta sin ella y marca
          el modelo para no reenviarla en el resto de la ejecución (el muestreo lo
          controla 'effort')."""
        if getattr(self, "_temp_unsupported", False) or not _sdk_accepts_temperature():
            api_kwargs.pop("temperature", None)
        try:
            return client.messages.create(**api_kwargs)
        except TypeError:
            # SDK viejo: output_config/thinking no estan en la firma -> extra_body.
            # SDK >= 1.0.0: 'temperature' ya no existe -> se cae, no se reenvia.
            extra = {}
            kwargs = dict(api_kwargs)
            for k in ("output_config", "thinking"):
                if k in kwargs:
                    extra[k] = kwargs.pop(k)
            if kwargs.pop("temperature", None) is not None:
                self._temp_unsupported = True
                logs.append("[INFO] El SDK anthropic instalado (>=1.0.0) no admite "
                            "'temperature'; se omite en el resto del batch.")
            return client.messages.create(extra_body=extra, **kwargs) if extra \
                else client.messages.create(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "temperature" in api_kwargs and "temperature" in msg \
                    and ("deprecat" in msg or "not supported" in msg
                         or "unsupported" in msg or "unexpected keyword argument" in msg):
                self._temp_unsupported = True
                kwargs = dict(api_kwargs)
                kwargs.pop("temperature", None)
                logs.append("[INFO] El modelo no admite 'temperature' (deprecada); se omite (control por 'effort').")
                return client.messages.create(**kwargs)
            raise

    @staticmethod
    def _extract_text(response):
        """Concatena los bloques type=='text' de la respuesta. Robusto a bloques
        thinking (que con adaptive pueden preceder al texto)."""
        parts = []
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", "") or "")
        return "\n".join(parts)

    # ----------------------------------------------------------

    def _image_to_base64_downscaled(self, img_path, max_side):
        """Carga, reduce el lado largo a max_side (manteniendo aspect) y codifica
        en base64. JPEG para fotos; PNG si tenía transparencia."""
        with Image.open(img_path) as img:
            has_alpha = img.mode in ("RGBA", "LA", "P")
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGBA" if has_alpha else "RGB")

            w, h = img.size
            longest = max(w, h)
            if longest > max_side:
                scale = max_side / float(longest)
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

            buf = io.BytesIO()
            if has_alpha:
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                img.save(buf, format="PNG")
                media_type = "image/png"
            else:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(buf, format="JPEG", quality=90)
                media_type = "image/jpeg"
            return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), media_type


# ============================================================
# REGISTRO
# ============================================================

NODE_CLASS_MAPPINGS = {
    "ClaudeDatasetProfiler": ClaudeDatasetProfiler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClaudeDatasetProfiler": "Claude Dataset Profiler (Rafa)",
}
