"""
claude_caption_generator.py
Nodo ComfyUI — Captioner (segundo nodo del sistema de captioning post-shift).
Parte del repo comfyui-rafa-nodes — github.com/osuvense/comfyui-rafa-nodes

Refactor post paradigm-shift (20 may 2026). Captiona imagen a imagen a resolución
plena bajo el paradigma de ENMASCARADO: describe lo que varía en el dataset, deja
sin describir los rasgos invariantes (el trigger los absorbe). Consume el perfil
validado del Dataset Profiler (invariant_traits + canonical_vocabulary) o esos
campos rellenados a mano por el operador.

Conserva la arquitectura agnóstica del nodo original (§ 2.1): input por carpeta,
iteración interna, skip_existing, log por imagen, model string libre, temperature,
api_key por env, outputs last_caption + log, output_dir/save_captions/max_images,
extra_instructions.

Derogado frente al pre-shift (§ 2.2): formato FLUX Dual / ZIT Prose, caption_length
short/medium/long, subject_description, NSFW_MODIFIER viejo (empujaba a describir
invariantes), y el system_prompt de override completo (la flexibilidad la da ahora
el modo Custom + extra_instructions; reañadirlo como escape hatch es decisión
abierta de Rafa — § 6).

Diseño: [REF]-nodo-captioning.md § 5.2, § 5.3, § 5.8, § 6.
"""

import os
import io
import base64
import re
import anthropic
from PIL import Image

try:
    from comfy.model_management import processing_interrupted
    COMFY_INTERRUPT_AVAILABLE = True
except ImportError:
    COMFY_INTERRUPT_AVAILABLE = False

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Los 5 modos confirmados (D7). Deben coincidir con los del Dataset Profiler.
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
# 'adaptive' va primero = default del combo (decisión producción 3 jun: observa mejor el
# detalle real). En Opus 4.8/4.7 temperature se ignora en ambos (deprecada). Fix #5.
THINKING_MODES = ["adaptive", "disabled"]

# ============================================================
# SYSTEM PROMPT — plantilla principal (v1, 29 may 2026 — § 5.8)
# Slots: {trigger_word} {mode_block} {invariant_traits}
#        {canonical_vocabulary} {nsfw_modifier} {extra_instructions}
#        {min_words} {max_words}
# Se rellena por .replace() (no .format()) para no chocar con llaves literales.
# ============================================================

CAPTIONER_SYSTEM_TEMPLATE = """You are the Captioner, the second of two chained stages in a LoRA-training
caption system for ComfyUI. You receive ONE image at full resolution and write
ONE training caption for it, in English, under the masked-captioning paradigm.
You also receive, from an upstream human-validated Dataset Profile, what to mask
and which scene vocabulary to keep consistent.

Trigger word for this dataset: {trigger_word}

## Paradigm: masked captioning

Whatever you DESCRIBE is excluded from what the LoRA learns; the model treats it
as general knowledge. Whatever you LEAVE UNDESCRIBED is bound to the trigger word
as an indivisible block. Your caption is a mask: describe what VARIES across the
dataset (so the LoRA does not fix it as identity), and leave the INVARIANT traits
undescribed (so the trigger absorbs them). Anchoring also works by lexical
repetition: a phrase repeated identically across the dataset gets anchored even
if it is not the trigger — so use the canonical vocabulary for stable scene
elements, but genuinely vary your wording for things that change per image.

## Describe vs do not describe

ALWAYS DESCRIBE (variable, or controllable by prompt at inference):
- Scene and background.
- Clothing and variable accessories actually visible.
- Pose, gesture, posture.
- Lighting (type, source, hardness).
- Framing, camera angle, shot type.
- Props and objects in the scene.
- Facial expression.
- Medium — only if it varies across the dataset. If every image is the same
  medium it is invariant; do not describe it.

NEVER DESCRIBE (generic, non-content):
- Aesthetic quality, resolution, sharpness.
- Photographic meta / EXIF / camera model.
- Subjective or emotional language.

NEVER DESCRIBE (invariant identity):
- Everything in INVARIANT TRAITS below. The trigger carries them.

Hard wording rule: never use "young", "younger", "young man" or any variant. The
subject is the trigger; you rarely need a generic noun for the person, but if one
is unavoidable use "man", "adult man" or "mature man".

## Observation rules (apply to every image — observe before you write)

1. Jewelry checklist. Scan deliberately before writing: neck, each hand, each
   wrist, ears. Variable jewelry actually present must be described.

2. Expression under a moustache. The moustache hides the mouth — read BOTH the
   corners of the mouth AND the eye line. Corners up = smile. Corners down or
   straight + squinted eyes + tension = scowl/frown. Squinted eyes alone are NOT
   proof of a smile.

3. Clothing layers. Do not overlook secondary garments (an undershirt under an
   open shirt, a layer under a jacket).

4. No hallucinated objects. Describe only objects you actually see. Do not invent
   a cigarette, a drink, a phone.

5. No pareidolia. Do not read shadows as objects (a chin shadow is not a cord, a
   fold is not a strap).

6. No inference without observation. Assert something because you SEE it, not
   because the context suggests it.

7. Fine jewelry needs a confirmed contour. Never state a chain, cord or thin
   bracelet unless you can confirm a continuous contour along its whole path. A
   single point of glint is not enough — omit it.

8. Held object vs ring on a finger (critical). Ring = glint always on the same
   finger + a continuous band around the finger. Held object (lighter, key, small
   phone) = glint between closed fingers + no band + a rectangular or cylindrical
   shape. Do not confuse them.

9. Laterality from face direction (the most error-prone case). Read which way the
   face points first, then derive the body:
   - Face points to the LEFT of the frame -> the subject rotated to THEIR right
     -> the LEFT side of their body is more visible -> their LEFT arm is at the
     front. And vice versa.
   - Confirm face direction from several cues: the eye line, the direction of the
     nose by its base, the visible ear (if only one ear shows, it is on the side
     OPPOSITE to where the face points), the chin line.
   - Do NOT use the moustache as a directional cue — it is symmetric.
   - In high doubt, describe agnostically ("one hand", "the other arm", "on the
     left side of the frame").
   - Distinguish "looking toward the camera" (pupils centered or nearly) from
     "looking outward / out of frame" (the default when the eyes point to a side
     of the frame, not at the viewer).

10. Ring finger identity. To name the finger a ring is on, count fingers from the
    thumb.

11. No over-specification. Describe visible effects, do not attribute causes (no
    invented "color cast", no invented reason for the light). Do not invent
    subtle decorative details you are not sure are there.

12. Evidence calibration (three levels). No evidence -> omit. Reasonable evidence
    -> describe conservatively. Clear evidence -> describe in detail. "No jewelry
    without a confirmed contour" means "omit the UNCONFIRMED", not "omit
    everything slightly doubtful".

13. Pose before background (critical). FIRST determine the pose from the
    subject's BODY alone (torso, legs, head) — never from the background. THEN
    identify background elements independently. Verify they are coherent, but
    never derive the pose from a background object you think you recognized.

14. Nail finish — do not invent. Treat nails as natural by default. Do not say
    nails are painted, polished, glossy, "pale-polished" or colored, and do not
    assign any nail color or finish, unless a colored coating is unambiguously
    visible with a clear colored edge. A glint or a pale tone on a nail is NOT
    polish — omit it. This rule is ONLY about nails; it does not restrict
    describing real scene colors, lighting tints or surfaces, which are variable
    context and must still be described.

## Mode for this dataset

{mode_block}

## Invariant traits for this dataset (do NOT describe these)

{invariant_traits}

These are masked: the trigger absorbs them. Do not name them, describe them, or
allude to them — not even indirectly.

## Canonical scene vocabulary (use these exact descriptors for consistency)

{canonical_vocabulary}

Use the canonical phrasing verbatim for the elements it covers. Do NOT apply
canonical phrasing to anything that should stay variable per image.

## NSFW

{nsfw_modifier}

## Extra instructions for this session

{extra_instructions}

## Output format

Output ONLY the caption for the single image provided — nothing else. No preamble,
no quotation marks, no markers, no commentary, no translation. The node saves your
entire output verbatim as the image's .txt caption.

Caption format:
- Begin with the trigger word, then a space, then the description, with no
  separating period after the trigger.
- Natural, fluent English prose. No keyword block, no lists, no dual-hybrid
  format, no bullet points.
- {min_words}-{max_words} words.
- Describe only the variable content per the rules above; mask the invariant
  traits.
- Do not start with "a photo of" or "this image shows". Do not mention these
  instructions."""


# Bloques de modo — se inyecta uno en {mode_block} (§ 5.8)
MODE_BLOCKS = {
    "Identidad-persona": """Mode: Identidad-persona. This dataset teaches one person's identity. The person's
permanent appearance is invariant (see invariant traits) and must be masked — the
trigger carries it. Describe only what varies image to image: scene, clothing,
pose, expression, lighting, framing, props. Nudity is context, not identity: mark
it with "nude" or "shirtless" as a variable state, but do not describe the body or
genitals themselves (invariant). Tone: neutral, factual.""",

    "Anatomía-sujeto-específico": """Mode: Anatomía-sujeto-específico. This dataset teaches ONE specific subject's
anatomy. The subject's PERMANENT anatomy is invariant (see invariant traits) and is
masked — the trigger carries its identity (characteristic shape, proportions and
identifying features). Do not describe those identifying traits.
Describe only what VARIES, exactly as a person's facial EXPRESSION is described while
the face itself stays masked. The variable state here is mechanical: arousal state
(erect / semi-erect / flaccid), foreskin position as a state (retracted exposing the
glans / covering the glans), hand grip, contact, position, camera angle, lighting,
background. Describe these clinically and explicitly (the NSFW block applies).
The line to hold: STATE is variable and IS described (it is controllable at inference,
like a smile is a state of the same mouth); IDENTITY is invariant and is masked.
Naming the glans or foreskin only to state its current position is describing a STATE
— allowed. Describing the penis's characteristic size, shape or identity is describing
IDENTITY — masked. Never attach identity descriptors (size, "large", distinctive
shape) to the state. Tone: clinical, factual.""",

    "Concepto-acción-pose": """Mode: Concepto-acción-pose. This dataset teaches a concept, action or pose across
DIFFERENT subjects. Here anatomy is NOT invariant (it changes per subject) — so
describe it, to make the LoRA learn the concept and not one instance. Describe what
defines the concept (the action, pose, configuration, interaction) clinically and
precisely. Mask only what the invariant list declares constant across the set.
Tone: clinical, factual.""",

    "Estilo": """Mode: Estilo. This dataset teaches a visual style. The style is invariant and is
carried by the trigger — do not describe it. Describe the CONTENT instead (subject,
scene, composition, what is depicted) so the LoRA binds the style, not the content,
to the trigger. Mask only what the invariant list declares. Tone: neutral, factual.""",

    "Custom": """Mode: Custom. Follow the operator-supplied definition of what to mask and what to
describe (given in the invariant traits and extra instructions). Apply every
universal rule above against that definition.""",
}

# Modificador NSFW — se inyecta uno en {nsfw_modifier} (§ 5.8)
NSFW_ON = """NSFW is enabled. You may describe nudity, sexual content, arousal states, acts and
positions explicitly and clinically — but ONLY for elements that VARY across the
dataset. Anything in the invariant traits stays masked even with NSFW on. In
Identidad-persona, genitals are invariant: mark nudity as context (nude, shirtless)
without describing the genitals. In Anatomía-sujeto-específico and
Concepto-acción-pose, describe the variable STATE explicitly (arousal, foreskin
position retracted vs covering the glans, acts, positions) while still masking the
subject's invariant identity: state is variable and described, identity is masked.
Plain anatomical language; no euphemism, no subjective or arousing framing."""

NSFW_OFF = """NSFW is disabled. Do not produce explicit sexual description. If the image is
explicit, describe it only at a non-graphic, contextual level ("nude",
"shirtless") consistent with the masking rules."""


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


def _temp_via_extra_body(api_kwargs: dict) -> dict:
    """SDK >= 1.0.0: `temperature` ya no esta en la firma de messages.create(),
    pero la API la SIGUE honrando en los modelos anteriores al cambio; la guia
    oficial de migracion v1 manda enviarla por `extra_body` (ejemplo literal con
    claude-sonnet-4-6). Se hace solo si el SDK no la admite y el modelo no la ha
    deprecado; si la API acaba rechazandola, el retry la borra de extra_body."""
    t = api_kwargs.pop("temperature", None)
    if t is None:
        return api_kwargs
    eb = dict(api_kwargs.get("extra_body") or {})
    eb.setdefault("temperature", t)
    api_kwargs["extra_body"] = eb
    return api_kwargs


def _strip_temp(kwargs: dict) -> dict:
    """Quita temperature esta donde este (kwarg nativo o extra_body)."""
    kwargs = dict(kwargs)
    kwargs.pop("temperature", None)
    eb = kwargs.get("extra_body")
    if isinstance(eb, dict) and "temperature" in eb:
        eb = {k: v for k, v in eb.items() if k != "temperature"}
        if eb:
            kwargs["extra_body"] = eb
        else:
            kwargs.pop("extra_body", None)
    return kwargs


class ClaudeCaptionGenerator:
    """
    Captioner post-shift. Captiona imagen a imagen bajo el paradigma de enmascarado,
    consumiendo el perfil del Dataset Profiler (o invariant_traits / canonical_vocabulary
    rellenados a mano). Conserva toda la arquitectura agnóstica del nodo original.
    """

    CATEGORY = "rafa"
    OUTPUT_NODE = True
    FUNCTION = "generate_captions"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("last_caption", "log")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_folder": ("STRING", {
                    "default": "/workspace/datasets/MiPersonaje/raw",
                    "multiline": False,
                    "tooltip": "Ruta de la carpeta con las imágenes. Se procesan jpg, jpeg, png, webp, bmp."
                }),
                "trigger_word": ("STRING", {
                    "default": "MyCharacter",
                    "multiline": False,
                    "tooltip": "Trigger word del LoRA. El modelo lo antepone a cada caption."
                }),
                "caption_mode": (CAPTION_MODES, {
                    "tooltip": (
                        "Tipo de objetivo del dataset. Fija qué se prioriza y qué se enmascara por "
                        "defecto. Debe coincidir con el caption_mode del Dataset Profiler."
                    )
                }),
                "nsfw": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Activa descripción explícita de lo VARIABLE; los invariantes siguen enmascarados igual."
                }),
                "min_words": ("INT", {
                    "default": 35,
                    "min": 5,
                    "max": 300,
                    "step": 1,
                    "tooltip": "Mínimo de palabras del caption (post-shift: 35-80)."
                }),
                "max_words": ("INT", {
                    "default": 80,
                    "min": 5,
                    "max": 300,
                    "step": 1,
                    "tooltip": "Máximo de palabras del caption (post-shift: 35-80)."
                }),
                "model": ("STRING", {
                    "default": "claude-opus-4-8",
                    "tooltip": (
                        "Model string de Anthropic. Default claude-opus-4-8 (observación superior en la "
                        "validación, §5.9). Ejemplos: claude-opus-4-8, claude-sonnet-4-6. "
                        "Editable — actualizar cuando salgan modelos nuevos."
                    )
                }),
                "temperature": ("FLOAT", {
                    "default": 0.20,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "0.20 estable (no subir de 0.40). OJO: Opus 4.8/4.7 IGNORAN temperature (deprecada); en esos modelos el muestreo lo controla 'effort' y el nodo la omite solo."
                }),
                "max_images": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "tooltip": "0 = todas. 1 = modo prueba (solo la primera imagen)."
                }),
            },
            "optional": {
                "dataset_profile": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Perfil del Dataset Profiler (conéctalo de nodo a nodo o pega su texto). "
                        "Se usa SOLO si invariant_traits / canonical_vocabulary están vacíos: de él se "
                        "extraen los invariantes Confident y el vocabulario canónico."
                    )
                }),
                "invariant_traits": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Rasgos invariantes a enmascarar en ESTE dataset (lo que NO se describe). "
                        "Si está relleno, MANDA sobre el perfil. Vacío + perfil → se rellena con los "
                        "invariantes Confident del perfil."
                    )
                }),
                "canonical_vocabulary": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Descriptores canónicos de escena para consistencia léxica. Si está relleno, "
                        "MANDA sobre el perfil. Vacío + perfil → se rellena con el vocabulario del perfil."
                    )
                }),
                "extra_instructions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Instrucciones por sesión sin tocar la plantilla base."
                }),
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
                        "adaptive = el modelo decide cuánto razonar (DEFAULT de producción; observa mejor el detalle real — re-test 3 jun). "
                        "Sube max_tokens y la latencia/coste. disabled = sin extended thinking, más rápido y barato. "
                        "En Opus 4.8/4.7 temperature se ignora en ambos casos (deprecada)."
                    )
                }),
                "prompt_caching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Cachea el system prompt (idéntico en todo el batch): se cobra una vez, no por imagen. "
                        "Surfacea cache_w/cache_r en el log."
                    )
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Vacío → guarda los .txt junto a las imágenes. Relleno → usa este directorio."
                }),
                "save_captions": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "False → genera pero no escribe .txt (modo preview)."
                }),
                "skip_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True → salta imágenes que ya tienen .txt, sin gastar tokens."
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
        # Nodo con side-effect (escribe .txt al disco). ComfyUI cachea por inputs y NO
        # re-ejecutaría en sucesivos Queue con los mismos valores -> parece que "no guarda".
        # Devolver NaN lo marca siempre como cambiado, forzando ejecución en cada Queue.
        return float("nan")

    def generate_captions(
        self,
        image_folder,
        trigger_word,
        caption_mode,
        nsfw,
        min_words,
        max_words,
        model,
        temperature,
        max_images=0,
        effort="high",
        thinking="adaptive",
        dataset_profile="",
        invariant_traits="",
        canonical_vocabulary="",
        extra_instructions="",
        prompt_caching=True,
        output_dir="",
        save_captions=True,
        skip_existing=True,
        api_key="",
    ):
        logs = []
        last_caption = ""
        self._temp_unsupported = False

        # ---- Validaciones previas ----
        if not os.path.isdir(image_folder):
            return ("", f"[ERROR] La carpeta no existe: {image_folder}")

        key = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return ("", "[ERROR] No se encontró ANTHROPIC_API_KEY (ni en el nodo ni como variable de entorno).")

        if min_words > max_words:
            min_words, max_words = max_words, min_words
            logs.append("[WARN] min_words > max_words: se intercambiaron.")

        # ---- Resolver invariantes y vocabulario (parámetro manda; perfil rellena) ----
        inv = invariant_traits.strip()
        voc = canonical_vocabulary.strip()
        if (not inv or not voc) and dataset_profile.strip():
            parsed_inv, parsed_voc = self._parse_profile(dataset_profile)
            if not inv:
                inv = parsed_inv
                if parsed_inv:
                    logs.append("[INFO] invariant_traits tomado del perfil (invariantes Confident).")
            if not voc:
                voc = parsed_voc
                if parsed_voc:
                    logs.append("[INFO] canonical_vocabulary tomado del perfil.")

        if not inv:
            inv = "(none specified)"
            if caption_mode in ("Identidad-persona", "Anatomía-sujeto-específico"):
                logs.append(
                    "[WARN] Sin invariant_traits en modo de identidad/anatomía: el LoRA podría no "
                    "anclar la identidad. Rellena el campo o conecta un perfil."
                )
        if not voc:
            voc = "(none specified)"

        # ---- Recopilar imágenes ----
        images = sorted([
            f for f in os.listdir(image_folder)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])
        if not images:
            return ("", f"[ERROR] No se encontraron imágenes en: {image_folder}")

        total = len(images)
        if max_images > 0:
            images = images[:max_images]
        logs.append(f"[INFO] {len(images)} de {total} imágenes a procesar en {image_folder} — modo {caption_mode}")

        save_dir = output_dir.strip() if output_dir.strip() else image_folder

        # ---- Construir system prompt ----
        sys_prompt = self._build_system_prompt(
            trigger_word, caption_mode, inv, voc, nsfw,
            extra_instructions, min_words, max_words
        )
        system_param = [{"type": "text", "text": sys_prompt}]
        if prompt_caching:
            system_param[0]["cache_control"] = {"type": "ephemeral"}

        # ---- Cliente Anthropic ----
        client = anthropic.Anthropic(api_key=key)

        # ---- Procesar imágenes ----
        for fname in images:

            if COMFY_INTERRUPT_AVAILABLE and processing_interrupted():
                logs.append("[INTERRUMPIDO] Proceso detenido por el usuario.")
                break

            fname_base = os.path.splitext(fname)[0]
            img_path = os.path.join(image_folder, fname)
            txt_path = os.path.join(save_dir, fname_base + ".txt")

            if skip_existing and os.path.exists(txt_path):
                logs.append(f"[SKIP] {fname}")
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        last_caption = f.read().strip()
                except Exception:
                    pass
                continue

            try:
                img_b64, media_type = self._image_to_base64(img_path)
            except Exception as e:
                logs.append(f"[ERROR] {fname} — fallo al cargar imagen: {e}")
                continue

            user_content = [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": img_b64},
                },
                {"type": "text", "text": "Write the single training caption for this image."},
            ]

            # Fix #5: thinking 'adaptive' es incompatible con temperature (doc Anthropic) ->
            # se omite temperature y se sube max_tokens (el razonamiento consume presupuesto);
            # 'disabled' respeta temperature y basta 400 tokens para un caption de 35-80 palabras.
            api_kwargs = dict(
                model=model.strip(),
                max_tokens=(4096 if thinking == "adaptive" else 400),
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
                caption = self._enforce_trigger(self._extract_text(response).strip(), trigger_word)
                usage = response.usage
                tok_in = usage.input_tokens
                tok_out = usage.output_tokens
                cache_w = getattr(usage, "cache_creation_input_tokens", 0) or 0
                cache_r = getattr(usage, "cache_read_input_tokens", 0) or 0
            except Exception as e:
                logs.append(f"[ERROR] {fname} — API: {e}")
                continue
            if not caption:
                logs.append(f"[WARN] {fname} — respuesta sin bloque de texto (¿solo thinking?), no guardado.")
                continue

            if save_captions:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                    logs.append(f"[OK] {fname} — in:{tok_in} out:{tok_out} cache_w:{cache_w} cache_r:{cache_r}")
                except Exception as e:
                    logs.append(f"[WARN] {fname} — caption generado pero no guardado: {e}")
            else:
                logs.append(f"[OK] {fname} (no guardado) — in:{tok_in} out:{tok_out} cache_w:{cache_w} cache_r:{cache_r}")

            last_caption = caption

        return (last_caption, "\n".join(logs))

    # ----------------------------------------------------------

    def _build_system_prompt(
        self, trigger_word, caption_mode, invariant_traits,
        canonical_vocabulary, nsfw, extra_instructions, min_words, max_words
    ):
        mode_block = MODE_BLOCKS.get(caption_mode, MODE_BLOCKS["Identidad-persona"])
        nsfw_modifier = NSFW_ON if nsfw else NSFW_OFF
        extra = extra_instructions.strip() or "(none)"
        trigger = trigger_word.strip() or "TRIGGER"

        out = CAPTIONER_SYSTEM_TEMPLATE
        out = out.replace("{trigger_word}", trigger)
        out = out.replace("{mode_block}", mode_block)
        out = out.replace("{invariant_traits}", invariant_traits)
        out = out.replace("{canonical_vocabulary}", canonical_vocabulary)
        out = out.replace("{nsfw_modifier}", nsfw_modifier)
        out = out.replace("{extra_instructions}", extra)
        out = out.replace("{min_words}", str(min_words))
        out = out.replace("{max_words}", str(max_words))
        return out

    # ----------------------------------------------------------

    def _parse_profile(self, profile_text):
        """Extrae del Dataset Profile (Markdown del Profiler) los invariantes
        Confident y el bloque de vocabulario canónico. Vía exprés Profiler->Captioner
        sin nodo de pausa: solo los Confident se enmascaran; los Borderline NO
        (decisión del operador). § 6, revisión 3."""
        lines = profile_text.splitlines()

        # --- Invariantes Confident ---
        inv_items = []
        in_confident = False
        for line in lines:
            stripped = line.strip()
            low = stripped.lower()
            if low.startswith("confident"):
                in_confident = True
                continue
            if in_confident:
                # Fin del bloque Confident
                if low.startswith("borderline") or stripped.startswith("#") \
                        or low.startswith("### variable") or low.startswith("variable traits"):
                    break
                if stripped.startswith("-"):
                    item = stripped.lstrip("-").strip()
                    # Quitar la nota tras el primer guión largo / " - " / ":"
                    for sep in ("—", " – ", " - ", ":"):
                        if sep in item:
                            item = item.split(sep, 1)[0].strip()
                            break
                    if item:
                        inv_items.append(item)
                elif stripped == "":
                    continue
        invariants = "\n".join(f"- {it}" for it in inv_items)

        # --- Vocabulario canónico (sección entera) ---
        voc_lines = []
        in_voc = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("###") and "canonical scene vocabulary" in stripped.lower():
                in_voc = True
                continue
            if in_voc:
                if stripped.startswith("###") or stripped.startswith("## "):
                    break
                voc_lines.append(line.rstrip())
        vocabulary = "\n".join(voc_lines).strip()

        return invariants, vocabulary

    # ----------------------------------------------------------

    @staticmethod
    def _enforce_trigger(caption, trigger_word):
        """Garantiza que el caption empieza con el trigger EXACTO (case incluido).
        Fix #2a (3 jun 2026): el modelo escribió 'YUM' en vez de 'Yum'; no dependemos de
        que respete el case. Si ya empieza por el trigger (ignorando case), se normaliza
        ese prefijo al canónico conservando el resto; si no, se antepone el trigger."""
        trig = (trigger_word or "").strip()
        if not trig:
            return caption
        cap = caption.lstrip()
        if not cap:
            return trig
        if cap[:len(trig)].lower() == trig.lower():
            rest = cap[len(trig):]
            # Boundary check (fix 10 jun 2026): "Yummy treats" NO empieza por el
            # trigger "Yum" — tras el prefijo debe venir fin de texto, espacio o
            # puntuacion (no otra letra/digito); si no, se antepone el trigger.
            if not rest or not rest[0].isalnum():
                return trig + rest
        return trig + " " + cap

    def _create_with_fallback(self, client, api_kwargs, logs):
        """Llama a messages.create con dos salvaguardas:
        - Si el SDK no acepta output_config/thinking como kwargs nativos (versión
          < 0.105.2), los reenvía vía extra_body.
        - Si el modelo ha deprecado 'temperature' (Opus 4.8/4.7 devuelven 400
          '`temperature` is deprecated for this model.'), reintenta sin ella y marca
          el modelo para no reenviarla en el resto del batch (control por 'effort')."""
        api_kwargs = dict(api_kwargs)
        if getattr(self, "_temp_unsupported", False):
            api_kwargs.pop("temperature", None)
        elif not _sdk_accepts_temperature():
            # SDK >= 1.0.0: fuera de la firma, pero la API la honra en modelos
            # anteriores al cambio -> se envia por extra_body (MIGRATION.md v1).
            api_kwargs = _temp_via_extra_body(api_kwargs)
        had_temp = "temperature" in api_kwargs or \
            "temperature" in (api_kwargs.get("extra_body") or {})
        try:
            return client.messages.create(**api_kwargs)
        except TypeError as e:
            # SDK viejo: output_config/thinking no estan en la firma -> extra_body.
            extra = dict(api_kwargs.get("extra_body") or {})
            kwargs = dict(api_kwargs)
            kwargs.pop("extra_body", None)
            for k in ("output_config", "thinking"):
                if k in kwargs:
                    extra[k] = kwargs.pop(k)
            if "temperature" in str(e).lower() and "temperature" in kwargs:
                # el TypeError venia de temperature: no se reenvia como kwarg
                extra.setdefault("temperature", kwargs.pop("temperature"))
            return client.messages.create(extra_body=extra, **kwargs) if extra \
                else client.messages.create(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            if had_temp and "temperature" in msg \
                    and ("deprecat" in msg or "not supported" in msg
                         or "unsupported" in msg or "unexpected keyword argument" in msg):
                self._temp_unsupported = True
                kwargs = _strip_temp(api_kwargs)
                logs.append("[INFO] El modelo no admite 'temperature' (deprecada); se omite en el resto del batch (control por 'effort').")
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

    # Lado largo maximo que conserva la API de Anthropic: imagenes mayores se
    # reescalan en el servidor a ~1568px (los tokens se cobran sobre el tamano
    # reescalado) y >5MB/8000px devuelven 400. Reescalar en local da el mismo
    # resultado de observacion, elimina el riesgo de rechazo y acelera el upload.
    # Fix 10 jun 2026 (el Profiler ya muestreaba a 768; esto es el simil a full-res).
    API_MAX_SIDE = 1568

    def _image_to_base64(self, img_path):
        ext = os.path.splitext(img_path)[1].lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/png",
        }
        media_type = media_type_map.get(ext, "image/jpeg")

        with Image.open(img_path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            w, h = img.size
            longest = max(w, h)
            if longest > self.API_MAX_SIDE:
                scale = self.API_MAX_SIDE / float(longest)
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            buf = io.BytesIO()
            save_format = "PNG" if media_type == "image/png" else "JPEG"
            if save_format == "JPEG":
                img.save(buf, format="JPEG", quality=92)
            else:
                img.save(buf, format="PNG")
            return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), media_type


# ============================================================
# REGISTRO
# ============================================================

NODE_CLASS_MAPPINGS = {
    "ClaudeCaptionGenerator": ClaudeCaptionGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClaudeCaptionGenerator": "Claude Caption Generator (Rafa)",
}
