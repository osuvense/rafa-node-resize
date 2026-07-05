# -*- coding: utf-8 -*-
"""
claude_prompt_generator.py
Meganodo de generacion de prompts via Claude API.
Parte del repo comfyui-rafa-nodes - github.com/osuvense/comfyui-rafa-nodes

Refactor paradigm-shift-aware (jun 2026). Tres ejes nuevos sobre el nodo original:

1) MODO (dropdown):
   - "LoRA solo"                 -> usa los toggles de LoRA; el paradigma lo da cada
                                    LoRA (preshift describe todo / postshift enmascara).
                                    NO inyecta taste profile.
   - "Improvisacion sin LoRA"    -> ignora toggles; el LLM monta el prompt desde una
                                    idea vaga + el taste profile (si esta activo).
                                    Para probar modelos a pelo sin LoRA.
   - "LoRA + improvisacion"       -> triggers de LoRA + taste profile combinados.

   LoRAs: ceylan/lexte/yum son preshift ZIT (se describe todo con su vocabulario).
   ceyblan es POST-SHIFT (mismo personaje que ceylan, renombrado): su trigger absorbe
   la identidad -> por defecto el nodo NO describe rasgos invariantes (calva, barriga,
   vello, bigote, PIES, genitales), solo lo variable (escena, pose, ropa, luz, camara).

   IDENTITY_BOOST (4 jun 2026, solo LoRAs post-shift): el enmascarado es regla de
   TRAINING; en inferencia, si el binding del trigger es debil (LoRA underfit, ej.
   CeyblanKLEIN v1), reforzar nombrando es el workaround estandar de la comunidad.
   Niveles: off (estricto, trigger + escena) / class_word ("Ceyblan man") /
   light (class word + 2-3 rasgos clave) / full (descripcion completa pre-shift).

2) MODELO DESTINO (dropdown): cambia las reglas de prompting que se le dan al LLM.
   - Z-Image Turbo  : prosa single-encoder Qwen3-4B (lo original).
   - Klein / FLUX.2 : prosa single-encoder Qwen3-8B, CFG real, usa negative.
   - FLUX.1 legacy  : DUAL encoder -> rellena clip_l + t5xxl por separado.
   - Chroma1-HD     : prosa + tags de calidad, usa negative.

3) TASTE PROFILE embebido + toggle: ADN estetico destilado de los captions de
   produccion. Activable/desactivable. Solo actua en los modos de improvisacion.

CAPTION DIGESTS EMBEBIDOS (10 jun 2026) — sustituyen la carga de captions de disco.
El nodo pre-shift cargaba los corpus de training ENTEROS (claude_context_*.txt,
hasta ~45k tokens con Lesty) en cada llamada: coste disparado y ruido que el LLM no
aprovechaba. Ahora cada LoRA pre-shift lleva en LORA_DOCS un digest medido de su
corpus real (anclas lexicas con tasas de presencia, estructura del caption tipo,
cobertura del dataset y sesgos a compensar), destilado de los .md canonicos del
project ([LEGACY]-CeylanV5-captions.md / [LEGACY]-LestyV3-captions.md /
[LEGACY]-YumV3-captions.md) con analisis de frecuencias (10 jun 2026). Sin carga de
disco: portabilidad total.

HARDENING API (10 jun 2026, post-incidente 7-8 jun):
- Prompt caching de Anthropic SIEMPRE activo: el system prompt se parte en bloque
  ESTABLE (core + guia de modelo + taste + docs de LoRAs; cache_control ephemeral)
  y bloque VARIABLE (dials + formato). Re-llamadas con los mismos toggles pagan el
  bloque gordo a ~0.1x.
- Fallback de `temperature` deprecada (Opus 4.8/4.7 → 400): se reintenta sin ella
  (mismo patron que Captioner/Profiler, fix `4bc5100`).
- Extraccion de texto robusta a bloques thinking (`_extract_text`).
- Usage (in/out/cache_w/cache_r) en el log de consola por llamada.
- max_tokens escala con `variants` (4096 base / 8192 con 4+; subido 5 jul 2026, JSON truncado).

Compatibilidad: los outputs originales (prompt/razonamiento) conservan su posicion;
los nuevos (clip_l/t5xxl/negative) van al final. OJO (4 jun 2026): el reorden de
widgets (toggles BOOLEAN agrupados + identity_boost) descuadra los widgets_values de
workflows guardados con versiones anteriores -> re-marcar los valores una vez.

JSON obligatorio como output - mismo framing que el agente Telegram.
"""

import os
import json
import anthropic
from typing import Dict, Any, List, Tuple

# ============================================================
# TASTE PROFILE EMBEBIDO - ADN estetico de Rafa
# ------------------------------------------------------------
# Destilado de los captions de produccion (Lesty/Yum/Ceylan/Ceyblan) y del
# vocabulario de inferencia consolidado en [REF]-klein-stack.md.
# NOTA: esto es para INFERENCIA (describir el lote completo), NO para captions
# de training (donde se enmascara la identidad). En inferencia sin LoRA hay que
# describir todo explicitamente porque no hay trigger que absorba la identidad.
# Solo se inyecta en los modos de improvisacion y si el toggle esta activo.
# ============================================================

TASTE_PROFILE = """## SUBJECT AESTHETIC GUIDE (apply when improvising or when the user does not fully specify the man)

Default subject: a mature, very heavy adult man (morbidly obese / superchub), roughly 45-70 years old. Photoreal, candid, unidealized real body. NEVER glamorized, athletic or slim by default.

Core features to favor unless the user explicitly says otherwise:
- BUILD: very obese, heavyset, soft. A very large, protruding belly that hangs heavily downward and overhangs; the belly is a focal element of the image.
- BODY HAIR: densely hairy - thick hair on chest, belly, arms and legs; gray/silver patterns welcome.
- HEAD / FACE: bald head OR short gray hair; a thick gray or silver mustache, very often a full thick gray beard; visible double chin.
- SKIN: light or light-brown, mature texture, natural pores; not airbrushed.
- FEET: bare feet are a recurring point of interest; when feet are in frame, render them clearly and in detail.
- EXPLICIT MALE ANATOMY (when NSFW and in frame): uncircumcised, retracted foreskin; state flaccid / semi-erect / fully erect; pubic area covered in dense dark hair.

Preferred phrasings (reuse these exact descriptors for consistency):
"mature obese bear-build man", "large round prominent belly", "large protruding belly hanging heavily downward", "large hairy overhanging belly", "hairy chest and belly", "dense gray body hair", "bald head, thick gray mustache", "full thick gray beard", "in his late sixties", "completely nude" (not just "nude").

Tone: intimate, candid, documentary realism of a real heavy mature body. If a partner/second man appears and the user does not specify him, he matches the same prototype unless told otherwise.

This is a GUIDE, not a cage: always honor explicit user choices (a slimmer partner, specific clothing, a specific setting, SFW, etc.) over these defaults."""

# ============================================================
# DOCUMENTACION DE LORAS - embebida para portabilidad
# ------------------------------------------------------------
# 10 jun 2026: cada LoRA pre-shift incorpora su CAPTION DIGEST (destilado medido
# del corpus real de training, con tasas de presencia) en lugar de cargar los
# captions completos de disco. Fuente y metodo: [LEGACY]-*-captions.md del
# project + analisis de frecuencias. Las "reglas de produccion" (validadas en
# inferencia ZIT) se conservan y se etiquetan como tales cuando difieren del
# vocabulario literal del corpus.
# ============================================================

LORA_DOCS = {

"ceylan": """
## LoRA: CeylanV5ZIT
- Trigger word: Ceylan (siempre al inicio del prompt)
- Modelo: Z-Image Turbo (encoder Qwen3-4B, prosa directa sin separacion CLIP/T5)

### Descripcion fisica
Hombre maduro (40-50 anos), complexion muy obesa. Cabeza completamente calva (bald head).
Bigote grueso (thick moustache). Vello corporal denso en pecho, abdomen, brazos y piernas.
Barriga grande y prominente (large, protruding belly). Piel clara o marron clara.
Doble papada visible en planos cerrados.

### Caption digest — corpus real de training (50 captions JoyCaption, medido 10 jun 2026)

Estructura del caption tipo (replicarla da el mejor anclaje):
"Photograph of Ceylan, a middle-aged, obese man with a bald head, thick moustache, and hairy chest. He [pose/accion], wearing [ropa]. His large, protruding belly is prominently visible. The background features [escenario]. The lighting is natural, with soft shadows. The camera angle is [angulo], capturing a [plano]."

Anclas lexicas medidas (apariciones sobre 50 captions):
- "Photograph of Ceylan, a ..." — 37/50, apertura canonica
- "obese" 47 · "protruding belly" 45 · "thick moustache" 33 (grafia britanica domina el corpus; "mustache" solo 11 y casi nunca con "gray") · "bald head" 29 · "hairy chest" 27
- "heavy build / heavyset" 18 · "light brown skin" 11 · "very high level of obesity" 5
- Luz: "natural" 37 vs "artificial" 6 · "soft shadows" 16
- Camara: "slightly low" 19 (sesgo real del dataset) · eye-level 11 · close-up 21 + medium shot/close-up 21 · full body solo 3

Cobertura del dataset (lo que el LoRA conoce de verdad): exteriores soleados (25),
pool/playa/mar (17), interior con puerta blanca (7), couch (3); camisa/shirt (21),
swim trunks (11), tank top (6), sunglasses (12). Desnudo casi ausente del corpus
(nude/naked 7/50, "completely nude" 3/50).

### Sesgos a compensar (validado en produccion)
- 34/50 fotos son close-up → especificar "shows his full body from head to feet" si quieres full body
- Angulo bajo espontaneo (19/50 "slightly low") → anadir "captured from eye level" si quieres angulo neutro
- Interior con puerta blanca aparece espontaneamente → especificar contexto
- "completely nude" OBLIGATORIO para desnudo (el corpus apenas tiene desnudos: hay que forzarlo, no vale "nude" solo)
- NOTA: "hanging heavily downward" NO es vocabulario del corpus (0/50); es frase de inferencia validada en produccion ZIT. Funciona, pero el ancla de training real es "large, protruding belly".

### Reglas de prompt
1. "Ceylan" al inicio, solo
2. Mencionar siempre la barriga: "his large, protruding belly is prominently visible"
3. Estado de ropa siempre explicito
4. Angulo siempre explicito si importa
""",

"lexte": """
## LoRA: LexteV3ZIT
- Trigger word: Lexte (siempre al inicio, SIEMPRE seguido de "mature obese man, large belly")
- Modelo: Z-Image Turbo (encoder Qwen3-4B)
- Dataset: el de LestyV3 (336 captions Claude) con replace del trigger Lesty→Lexte en captions; el digest de abajo aplica integro.

### REGLA CRITICA — sesgo femenino
"Lexte" tiene asociacion fonética femenina en el corpus de entrenamiento.
SIN descriptores de genero explicitos, genera mujer.
FORMATO OBLIGATORIO: "Lexte, mature obese man, large belly, [resto del prompt]"
Nunca omitir "mature obese man, large belly" despues del trigger.

### Descripcion fisica
Hombre maduro, complexion obesa/bear-build. Pelo corto gris (salt-and-pepper).
Barba gris completa y espesa (full thick gray beard). Barriga grande y redonda.
Pecho y cuerpo densamente cubierto de vello corporal.
Tatuajes reales del corpus (vocabulario exacto, NO "colorful sleeve"):
- "a geometric polygon bear or panda face tattoo in bold black outline on his right upper pectoral"
- "a bear outline tattoo on his left upper arm and shoulder (sketch-line walking bears)"
- "a small cartoon bear tattoo on his lower back" (visible en poses de espalda)

### Caption digest — corpus real de training (336 captions Claude, medido 10 jun 2026)

Estructura del caption tipo:
"nsfw. Lexte, a mature obese bear-build man, [pose] [completely nude / ropa] [escenario]. He has short gray hair and a full thick gray beard. His hairy chest and large round prominent belly are clearly visible. [manos/props/anillo/pendiente]. [cierre: visibilidad de genitales + pies/dedos + tatuaje si visible]."

Anclas lexicas medidas (apariciones sobre 336 captions):
- Prefijo "nsfw." 259 / "sfw." 77 — TODOS los captions empiezan con ese marcador, ANTES del trigger. Usarlo en el prompt activa el registro correspondiente (es el ancla mas repetida del corpus).
- "a mature obese bear-build man" 311/336 — ancla absoluta tras el trigger
- "hairy chest" 209 · "completely nude" 183 (55% del corpus es desnudo) · "large round prominent belly" 148 · barba "full thick (gray/gray-dark) beard" ~220 con matiz variable · "short gray hair" 60 (variantes salt-and-pepper)
- Formula de cierre sistematica del corpus: visibilidad de genitales SIEMPRE declarada ("Genitals are not visible from this angle..." 70; "flaccid penis" 27 — el corpus NO contiene erecciones, 0) + pies y dedos ("toes" 165, "Feet are visible" 63) + "No tattoo is clearly visible in this image" o el tatuaje descrito en detalle.

Cobertura del dataset: cama/bed (89), exteriores (64), campo de trigo (40), couch (22),
studio (17), ducha/banyo (11), petalos de rosa (8); gorras/caps (138 — muy presentes),
underwear/briefs (40), jeans/pants (34), glasses (24), ear stud (18), anillos.
Poses: sentado (151), de pie (106), perfil/side (121), buttocks visibles (89), kneeling (27).
B/N: 63 captions (19% del corpus en black and white).

### Sesgos a compensar
- Corpus SOLO flacido → erecciones fuera de distribucion: no prometer detalle erecto con este LoRA.
- 19% del corpus es B/N → posible B/N espontaneo; para color asegurado, especificar "photographed in color".

### Reglas de prompt
1. "Lexte" al inicio, seguido INMEDIATAMENTE de "mature obese man, large belly"
2. Prefijo "nsfw." o "sfw." ANTES del trigger (patron del corpus)
3. Prosa descriptiva, no tags
4. Estado de ropa explicito
""",

"yum": """
## LoRA: YumV3ZIT
- Trigger word: Yum (siempre al inicio cuando esta activo)
- Modelo: Z-Image Turbo (encoder Qwen3-4B)
- Concepto: anatomia genital masculina (no es un personaje visual)

### Cuando usar Yum
- Primer plano genital (close-up de genitales)
- Cuando el pene es elemento principal del encuadre
- Cuando se necesita detalle anatomico especifico (ereccion, prepucio, venas, textura)

### Cuando NO usar Yum
- Poses de cuerpo completo donde los genitales no son el foco
- Retratos o planos de torso/cara
- Escenas SFW o topless

### Caption digest — corpus real de training (62 captions, medido 10 jun 2026)

Estructura del caption tipo:
"Yum. NSFW photograph of a close-up, top-down view of an uncircumcised with retracted foreskin, [fully erect / semi-erect / flaccid] penis with a [detalle del glans]. The penis is held by a hand with a firm grip near the base, fingers wrapped around the shaft. The skin is [tono] with visible veins. The surrounding area includes a large, hairy overhanging belly and dense pubic hair. The background is blurred. The lighting is artificial, casting soft shadows."

Anclas lexicas medidas (apariciones sobre 62 captions):
- "NSFW photograph" 62/62 — apertura universal ("Yum." inicial en 39; "Yum's" posesivo integrado en el resto)
- "uncircumcised with retracted foreskin" 62/62 LITERAL — usar tal cual, es el ancla anatomica del LoRA
- "close-up" 62/62 + "top-down" 61/62 — el encuadre ES el dataset; otros angulos = fuera de distribucion
- Estados: "semi-erect" 26 / "fully erect" 17 / "flaccid" 14 — declarar SIEMPRE uno
- "large overhanging belly" 49 (elemento contextual central) · agarre: grip* 94 ("fingers wrapped around" 28, "gripping the base" 24, "firm" 43) · "glans" 75 · "visible veins" 35 · scrotum/testicles 31
- Fondo "blurred" 57 · luz "artificial" 48 vs "natural" 12

### Reglas de prompt
1. "Yum" al inicio cuando esta activo
2. Describir siempre el estado de ereccion (flaccid / semi-erect / fully erect)
3. Mencionar siempre "large overhanging belly"
4. Especificar angulo: "close-up, top-down" es el nativo del dataset
""",

"ceyblan": """
## LoRA: Ceyblan (POST-SHIFT - paradigma enmascarado)
- Trigger word: "Ceyblan " al INICIO, con espacio, seguido de la descripcion SIN punto separador.
- Personaje: el MISMO que el "ceylan" pre-shift (Ceylan V5), renombrado en post-shift. No combinar con el toggle "ceylan".
- Modelos: CeyblanKLEIN (Klein/FLUX.2, ramal principal) o CeyblanZIT (Z-Image Turbo). Mismo trigger y mismo paradigma en ambos.

### REGLA CRITICA - enmascarado en inferencia
El trigger "Ceyblan" YA absorbe toda la identidad. NO describas sus rasgos invariantes:
- NO: edad, peso, calvicie, bigote, barba, vello facial/corporal, complexion, color de piel, color de ojos, estructura facial.
- NO: PIES (identidad permanente de este personaje; al reves que el resto - no los describas aunque esten en cuadro).
- NO: genitales (parte invariante; no se describen aunque sean visibles).
Redescribir eso pisa lo que el LoRA aporta.

### QUE SI describir (lo variable)
Escena y contexto, pose y accion, estado de ropa (marca solo "nude" / "shirtless" / la prenda), props y otras personas, iluminacion y atmosfera, camara (angulo, encuadre, plano, lente/film si fotorrealista).

### Formato
Prosa natural fluida en ingles, 35-80 palabras, sin keywords condensados. Empieza por "Ceyblan " + la escena.
Forma de ejemplo: "Ceyblan relaxing on a worn leather sofa in a dim living room, leaning back with one arm over the backrest, nude, warm side lamp light, shot at eye level, 35mm, candid photoreal."
""",

}

# Paradigma por LoRA: preshift (describir todo) vs postshift (el trigger absorbe la identidad)
LORA_PARADIGM = {
    "ceylan": "preshift",
    "lexte": "preshift",
    "yum": "preshift",
    "ceyblan": "postshift",
}

# Niveles del dial identity_boost (solo actua con LoRA post-shift activo).
# El enmascarado es regla de TRAINING; en inferencia, reforzar nombrando es el
# workaround estandar cuando el binding del trigger es debil (LoRA underfit).
IDENTITY_BOOST_LEVELS = ["off", "class_word", "light", "full"]

# Vocabulario de refuerzo por LoRA post-shift (reutiliza el vocabulario de
# inferencia validado en [REF]-klein-stack.md; extensible a futuros LoRAs).
LORA_BOOST_VOCAB = {
    "ceyblan": {
        "class_word": "man",
        "light": '"bald head", "thick gray mustache", "large protruding belly"',
        "full": ("mature obese man, completely bald head, thick gray mustache, "
                 "large protruding belly hanging heavily downward, hairy chest and belly, "
                 "dense body hair on arms and legs, double chin, light or light-brown skin; "
                 "explicit anatomy only when in frame: uncircumcised, retracted foreskin, "
                 "flaccid / semi-erect / fully erect"),
    },
}

# ============================================================
# GUIAS POR MODELO DESTINO
# Cada bloque le dice al LLM como escribir el prompt para ese modelo
# y que campos de salida rellenar.
# ============================================================

MODEL_GUIDES = {

"Z-Image Turbo": """## TARGET MODEL: Z-Image Turbo
Single text encoder (Qwen3-4B). Write ONE positive prompt in natural English prose,
direct and specific, about 2-4 sentences. No keyword lists, no CLIP/T5 split.
This model does NOT use a negative prompt: leave "negative" empty.
Leave "clip_l" and "t5xxl" empty.
Trigger words of active LoRAs go at the very start, comma-separated.""",

"Klein / FLUX.2": """## TARGET MODEL: FLUX.2 Klein
Single Qwen3-8B encoder, real CFG (CFGGuider @ ~4.5). Write ONE positive prompt in
natural English prose. Recommended structure:
[character triggers if any], [scene and general context], [detailed physical
description: body type, body hair, age, baldness/hair, mustache, beard], [specific
pose and action], [lighting and atmosphere], [photographic details: angle, framing,
sharpness, skin texture].
Word ORDER matters: put the most important elements first. Ideal length ~30-80 words.
For photorealism, name a camera/lens/film stock when it fits (e.g. "shot on Fujifilm
X-T5, 35mm f/1.4"). FLUX.2 supports neither keyword spam nor reliance on negation in
the positive prompt - describe what you WANT.
ALSO produce a "negative" prompt: Klein uses real CFG so negatives work, and they are
needed to fight the SNOFS merge female bias. Sensible baseline (adapt as needed):
"girly, femenine, vagina, pussy, blurry, out of focus, duplicated arms feet or fingers,
inconsistent positions, aberrations, unreal bodies".
Leave "clip_l" and "t5xxl" empty.""",

"FLUX.1 legacy": """## TARGET MODEL: FLUX.1 legacy (DUAL ENCODER)
This model uses TWO encoders. You MUST fill two SEPARATE fields, never the same text in both:
- "clip_l": trigger words + condensed keywords only (who, clothing state, NSFW/SFW).
  Shape example: "Ceylan Lesty, completely nude male, fully erect, full body, realistic photo, NSFW".
- "t5xxl": full descriptive prose. POSITION matters (relative positional embeddings):
  put each trigger word immediately before that character's descriptive block.
  Shape: "[scene context, lighting, angle], Trigger [facial descriptors], [body
  descriptors], [genital area if relevant], [final technical details]".
ALSO produce a "negative" prompt (quality + anatomy control).
Put a readable combined version of clip_l + t5xxl in "prompt" too (only for preview);
the fields that get wired into the workflow are "clip_l" and "t5xxl".""",

"Chroma1-HD": """## TARGET MODEL: Chroma1-HD
FLUX.1-schnell based foundational model. Write ONE positive prompt in natural English
prose, detailed. It also responds to quality/aesthetic tags: you MAY append tags such
as "aesthetic 11" or media tags like "photography_(artwork)" at the end when useful.
ALSO produce a "negative" prompt. Sensible baseline:
"low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged,
restricted palette, flat colors" (add the anti-female-bias terms if a man must be
guaranteed: "girly, femenine, vagina, pussy"). Guidance ~3.0.
Leave "clip_l" and "t5xxl" empty.""",

}

# Que campos espera cada modelo (para validacion/avisos)
MODEL_FLAGS = {
    "Z-Image Turbo":  {"dual": False, "negative": False},
    "Klein / FLUX.2": {"dual": False, "negative": True},
    "FLUX.1 legacy":  {"dual": True,  "negative": True},
    "Chroma1-HD":     {"dual": False, "negative": True},
}

# ============================================================
# NUCLEO COMUN DEL SYSTEM PROMPT
# ============================================================

SYSTEM_CORE = """Eres un agente experto en escribir prompts de generacion de imagen para los modelos de difusion que usa Rafa (trabajo adulto NSFW fotorrealista). Tu tarea: convertir la idea o escena del usuario - a veces muy vaga, a veces solo un par de palabras - en el o los prompt(s) optimos para el MODELO DESTINO indicado, alineados con su estetica, rellenando tu lo que el usuario no especifique.

REGLA DE SEGURIDAD ABSOLUTA (innegociable): NUNCA uses "young", "younger", "young man" ni variantes en el prompt de imagen. El sujeto es SIEMPRE un hombre adulto/maduro. Usa "man", "adult man", "mature man". Si necesitas contraste de edad o cuerpo, usa "slim adult man", nunca "young".

Contenido adulto: el contenido sexual explicito entre adultos es parte normal del trabajo; descripcion anatomica directa y clinica cuando la escena lo pida. Sin eufemismos innecesarios, sin moralizar.

USO DE LOS CAPTION DIGESTS: cuando un LoRA pre-shift esta activo, su digest te da las anclas lexicas REALES del training con sus tasas. Construye el prompt reutilizando esas frases exactas (mas tasa = mas anclaje) y respetando la estructura del caption tipo; no inventes sinonimos para los rasgos anclados.

REGLA DE FORMATO: responde UNICAMENTE con JSON valido. Sin texto antes ni despues. Sin markdown fences."""


def _as_bool(v) -> bool:
    """Acepta BOOLEAN nativo y strings legacy ('enabled'/'disabled')."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("enabled", "true", "1", "yes", "on")


def build_output_format(variants: int) -> str:
    """Bloque de FORMATO DE RESPUESTA del system prompt."""
    base = """FORMATO DE RESPUESTA (JSON, exactamente estas claves):
{
  "prompt": "prompt positivo principal en ingles (SIEMPRE relleno)",
  "clip_l": "solo FLUX.1 legacy; en otros modelos cadena vacia",
  "t5xxl": "solo FLUX.1 legacy; en otros modelos cadena vacia",
  "negative": "negative prompt en ingles si el modelo lo usa; si no, cadena vacia",
  "variants": [LISTA_DE_VARIANTES],
  "razonamiento": "explicacion BREVE en espanol de las decisiones tomadas"
}"""
    if variants > 1:
        base += f"""

Genera {variants} variantes DISTINTAS del prompt en el array "variants" (la primera
debe coincidir con "prompt"). Varia composicion, pose, encuadre, escenario y detalle;
manten coherente la estetica del sujeto. Para FLUX.1 legacy, cada variante es el texto
combinado de preview."""
    else:
        base += """

"variants" debe ser un array vacio []."""
    return base


# Constantes de los dropdowns
MODES = ["LoRA solo", "Improvisacion sin LoRA", "LoRA + improvisacion"]
TARGET_MODELS = ["Z-Image Turbo", "Klein / FLUX.2", "FLUX.1 legacy", "Chroma1-HD"]
NSFW_LEVELS = ["explicit", "suggestive", "sfw"]
FRAMINGS = ["auto", "portrait", "upper body", "full body", "genital close-up"]

SECTION_SEP = "\n\n" + "=" * 50 + "\n\n"


class ClaudePromptGenerator:
    """
    Meganodo de generacion de prompts via Claude API.
    Modo + modelo destino + taste profile + dials. Ver docstring del modulo.
    """

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = None
        self._temp_unsupported = False

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                # --- toggles de LoRA (BOOLEAN, agrupados; reorden 4 jun 2026) ---
                "ceylan": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "LoRA preshift ZIT (se describe todo con su vocabulario). "
                               "No combinar con 'ceyblan' (mismo personaje)."
                }),
                "lexte": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "LoRA preshift ZIT. Trigger con formato obligatorio anti-sesgo femenino."
                }),
                "yum": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "LoRA preshift ZIT de anatomia genital (concepto, no personaje)."
                }),
                "ceyblan": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "LoRA de identidad POST-SHIFT (mismo personaje que 'ceylan', renombrado): "
                               "su trigger absorbe la identidad. Refuerzo opcional via identity_boost. "
                               "No combinar con 'ceylan'."
                }),
                "scene": ("STRING", {
                    "default": "Describe la escena o suelta un par de ideas vagas.",
                    "multiline": True
                }),
                "mode": (MODES, {
                    "default": MODES[0],
                    "tooltip": "LoRA solo = toggles de LoRA sin taste profile. "
                               "Improvisacion = el LLM monta el prompt desde tu idea + taste profile. "
                               "Mixto = triggers de LoRA + taste profile."
                }),
                "target_model": (TARGET_MODELS, {
                    "default": TARGET_MODELS[0],
                    "tooltip": "Cambia las reglas de prompting. FLUX.1 legacy rellena clip_l + t5xxl."
                }),
                "taste_profile": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Inyecta tu ADN estetico embebido. Solo actua en los modos de "
                               "improvisacion. Desactivalo para probar el modelo 'limpio' sin sesgar."
                }),
                "identity_boost": (IDENTITY_BOOST_LEVELS, {
                    "default": "light",
                    "tooltip": "Solo actua con LoRA post-shift activo (ceyblan). Cuanto refuerza el "
                               "prompt la identidad: off = trigger + escena (ideal con LoRA bien "
                               "horneada); class_word = 'Ceyblan man'; light = class word + 2-3 "
                               "rasgos clave; full = descripcion completa estilo pre-shift. "
                               "Con CeyblanKLEIN v1 (floja): light o full."
                }),
                "nsfw": (NSFW_LEVELS, {
                    "default": "explicit",
                    "tooltip": "explicit = sexual directo; suggestive = insinuado; sfw = sin desnudo."
                }),
                "framing": (FRAMINGS, {
                    "default": "auto",
                    "tooltip": "Plano sugerido. 'auto' = el LLM decide segun la escena."
                }),
                "variants": ("INT", {
                    "default": 1, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Cuantas variantes generar de un tiro (van en el campo razonamiento)."
                }),
                "creativity": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Mapea a temperature. Bajo = fiel/estable; alto = improvisa mas. "
                               "OJO: Opus 4.8/4.7 deprecan temperature; el nodo la omite solo si el modelo la rechaza."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Cache buster. Fijo + sin otros cambios = usa cache, NO gasta tokens. "
                               "Sube el seed (o ponlo en randomize/increment) para forzar una variante nueva."
                }),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "claude_model": ("STRING", {
                    "default": "claude-sonnet-4-6",
                    "multiline": False,
                    "tooltip": "Modelo de Claude. Sonnet 4.6 por defecto (sobra para prompts; "
                               "Opus ~1.7x mas caro por token y deprecó temperature)."
                }),
                "extra_directives": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Instrucciones extra ad-hoc para esta generacion."
                }),
                "taste_profile_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Si no esta vacio, sustituye al taste profile embebido (sigue en codigo, "
                               "esto es solo un override puntual in-canvas)."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "razonamiento", "clip_l", "t5xxl", "negative")
    FUNCTION = "generate_prompt"
    CATEGORY = "rafa"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, scene="", mode="", target_model="", taste_profile=False,
                   ceylan=False, lexte=False, yum=False, ceyblan=False,
                   identity_boost="light", nsfw="", framing="",
                   variants=1, creativity=0.8, seed=0, claude_model="",
                   extra_directives="", taste_profile_override="", **kwargs):
        # Clave determinista de los inputs: el nodo solo se re-ejecuta (y gasta
        # tokens en la API) cuando cambia alguno. Repetir Queue con todo igual usa
        # la cache de ComfyUI y NO vuelve a llamar. Para forzar una variante nueva
        # sin tocar la escena, sube 'seed' (o ponlo en randomize/increment), igual
        # que el seed de un KSampler. api_key se excluye a proposito.
        import hashlib
        key = repr((scene, mode, target_model, _as_bool(taste_profile),
                    _as_bool(ceylan), _as_bool(lexte), _as_bool(yum), _as_bool(ceyblan),
                    identity_boost, nsfw, framing, int(variants),
                    round(float(creativity), 4), int(seed), claude_model,
                    extra_directives, taste_profile_override))
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    # ----------------------------------------------------------

    def _build_system_blocks(
        self,
        mode: str,
        target_model: str,
        active: Dict[str, bool],
        use_taste: bool,
        nsfw: str,
        framing: str,
        variants: int,
        extra_directives: str,
        taste_text: str,
        identity_boost: str = "light",
    ) -> Tuple[str, str]:
        """Construye el system prompt en DOS bloques para el prompt caching de
        Anthropic: (estable, variable).

        ESTABLE (cacheable; solo cambia al cambiar modo/modelo/toggles/taste):
        core + guia del modelo destino + taste profile + docs/digests de LoRAs.
        VARIABLE (pequeno, sin cache): dials nsfw/framing, reglas identity_boost,
        extra_directives y formato de salida.
        """
        stable: List[str] = [SYSTEM_CORE.strip()]
        variable: List[str] = []

        # --- ESTABLE 1) Guia del modelo destino
        stable.append(MODEL_GUIDES[target_model].strip())

        # LoRAs solo cuentan en los modos que los usan (en improvisacion pura los
        # toggles se ignoran del todo; fix 10 jun 2026 — antes la nota post-shift
        # del taste se colaba sin trigger ni docs).
        using_loras = mode in (MODES[0], MODES[2])
        postshift_active = using_loras and any(
            active.get(n) and LORA_PARADIGM.get(n) == "postshift" for n in active
        )

        # --- ESTABLE 2) Taste profile (solo en modos de improvisacion)
        improvising = mode in (MODES[1], MODES[2])
        if improvising and use_taste and taste_text.strip():
            taste_block = ("## YOUR AESTHETIC (default taste, apply unless overridden)\n"
                           + taste_text.strip())
            if postshift_active:
                taste_block += ("\n\nNOTE: an identity LoRA is active. Use this aesthetic ONLY to steer "
                                "mood, setting, framing and tone - do NOT use it to describe the character's "
                                "body, face, feet or genitals (the LoRA fixes those; see the post-shift rule).")
            stable.append(taste_block)

        # --- ESTABLE 3) Docs/digests de LoRAs activos
        if using_loras and any(active.values()):
            stable.append("## ACTIVE LoRAs AND THEIR DOCUMENTATION")
            for name, enabled in active.items():
                if enabled:
                    stable.append(LORA_DOCS[name].strip())
        elif using_loras and not any(active.values()) and mode == MODES[0]:
            # "LoRA solo" sin loras activos: comportamiento original (prompt generico)
            stable.append("No hay LoRAs activos. Genera un prompt generico basado en la escena.")

        # --- VARIABLE 1) Dials: NSFW + framing
        nsfw_map = {
            "explicit": "NSFW level: EXPLICIT. Explicit adult sexual content and direct "
                        "anatomical description are expected when the scene calls for it.",
            "suggestive": "NSFW level: SUGGESTIVE. Sensual/implied, nudity partial or "
                          "teased, no explicit genital action.",
            "sfw": "NSFW level: SFW. No nudity, no sexual content. Clothed scene.",
        }
        variable.append(nsfw_map[nsfw])
        if framing != "auto":
            variable.append(f"Preferred framing/shot: {framing}. Reflect it in the prompt.")

        # --- VARIABLE 2) Regla post-shift + identity_boost (depende del dial)
        if postshift_active:
            boost = identity_boost if identity_boost in IDENTITY_BOOST_LEVELS else "light"
            rule = [
                "## PARADIGMA POST-SHIFT ACTIVO (LoRA de identidad enmascarado)",
                "Hay un LoRA de identidad post-shift activo. Su trigger word aporta la identidad "
                "del personaje. Describe SIEMPRE lo variable: escena, pose, accion, estado de ropa "
                "(marca nude/shirtless o la prenda), props, otras personas, iluminacion y camara. "
                "El trigger va al inicio, con espacio, seguido de la descripcion.",
            ]
            if boost == "off":
                rule.append(
                    "REFUERZO DE IDENTIDAD: OFF (estricto). NO describas ningun rasgo invariante "
                    "del personaje (edad, peso, calva, barriga, vello, bigote, barba, complexion, "
                    "piel, ojos, estructura facial, PIES, genitales) ni anadas class word: el "
                    "trigger los absorbe. Solo trigger + escena.")
            elif boost == "class_word":
                rule.append(
                    "REFUERZO DE IDENTIDAD: CLASS_WORD. Inmediatamente despues del trigger anade "
                    "su class word (indicado abajo). NO describas rasgos fisicos del personaje "
                    "(tampoco pies ni genitales): el LoRA los aporta.")
            elif boost == "light":
                rule.append(
                    "REFUERZO DE IDENTIDAD: LIGHT. Tras el trigger anade su class word y refuerza "
                    "SOLO 2-3 rasgos clave usando el vocabulario exacto indicado abajo (los que "
                    "encajen con la escena). El resto de la identidad sigue enmascarada: NO "
                    "describas pies ni genitales ni mas rasgos.")
            else:  # full
                rule.append(
                    "REFUERZO DE IDENTIDAD: FULL. Tras el trigger + class word, describe la "
                    "identidad fisica completa del personaje con el vocabulario indicado abajo, "
                    "como con un LoRA pre-shift. Pies y anatomia explicita segun requiera la escena.")
            for n, enabled in active.items():
                if not (enabled and LORA_PARADIGM.get(n) == "postshift"):
                    continue
                vocab = LORA_BOOST_VOCAB.get(n, {})
                cw = vocab.get("class_word")
                if boost in ("class_word", "light", "full") and cw:
                    rule.append(f'[{n}] class word: "{cw}"')
                lv = vocab.get("light")
                if boost == "light" and lv:
                    rule.append(f"[{n}] rasgos clave (vocabulario exacto): {lv}")
                fv = vocab.get("full")
                if boost == "full" and fv:
                    rule.append(f"[{n}] vocabulario de identidad completo: {fv}")
            variable.append("\n".join(rule))

        # --- VARIABLE 3) Instrucciones extra ad-hoc
        if extra_directives.strip():
            variable.append("## INSTRUCCIONES EXTRA DE ESTA GENERACION\n" + extra_directives.strip())

        # --- VARIABLE 4) Formato de salida
        variable.append(build_output_format(variants))

        return (SECTION_SEP.join(stable), SECTION_SEP.join(variable))

    # ----------------------------------------------------------

    def _create_with_fallback(self, api_kwargs):
        """messages.create con la salvaguarda de `temperature` deprecada
        (Opus 4.8/4.7 → 400 '`temperature` is deprecated for this model.'):
        reintenta sin ella y no la reenvia el resto de la sesion del nodo.
        Mismo patron que Captioner/Profiler (fix `4bc5100`)."""
        if self._temp_unsupported:
            api_kwargs.pop("temperature", None)
        try:
            return self.client.messages.create(**api_kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "temperature" in api_kwargs and "temperature" in msg \
                    and ("deprecat" in msg or "not supported" in msg or "unsupported" in msg):
                self._temp_unsupported = True
                kwargs = dict(api_kwargs)
                kwargs.pop("temperature", None)
                print("[ClaudePromptGenerator] El modelo no admite 'temperature' (deprecada); se omite.")
                return self.client.messages.create(**kwargs)
            raise

    @staticmethod
    def _extract_text(response) -> str:
        """Concatena los bloques type=='text' de la respuesta. Robusto a bloques
        thinking que pudieran precederlos (mismo patron que Captioner/Profiler)."""
        parts = []
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", "") or "")
        return "\n".join(parts).strip()

    # ----------------------------------------------------------

    def generate_prompt(
        self,
        ceylan: bool,
        lexte: bool,
        yum: bool,
        ceyblan: bool,
        scene: str,
        mode: str,
        target_model: str,
        taste_profile: bool,
        identity_boost: str,
        nsfw: str,
        framing: str,
        variants: int,
        creativity: float,
        seed: int = 0,
        api_key: str = "",
        claude_model: str = "claude-sonnet-4-6",
        extra_directives: str = "",
        taste_profile_override: str = "",
    ) -> dict:

        if not scene.strip():
            raise ValueError("scene no puede estar vacio")

        key_to_use = api_key if api_key else self.api_key
        if not key_to_use:
            raise ValueError("No hay API key. Configura ANTHROPIC_API_KEY o pegala en el nodo.")

        if not self.client or api_key:
            self.client = anthropic.Anthropic(api_key=key_to_use)

        active = {
            "ceylan":  _as_bool(ceylan),
            "lexte":   _as_bool(lexte),
            "yum":     _as_bool(yum),
            "ceyblan": _as_bool(ceyblan),
        }

        use_taste = _as_bool(taste_profile)
        taste_text = taste_profile_override if taste_profile_override.strip() else TASTE_PROFILE

        stable_block, variable_block = self._build_system_blocks(
            mode=mode,
            target_model=target_model,
            active=active,
            use_taste=use_taste,
            nsfw=nsfw,
            framing=framing,
            variants=max(1, int(variants)),
            extra_directives=extra_directives,
            taste_text=taste_text,
            identity_boost=identity_boost,
        )

        # Prompt caching SIEMPRE activo: el bloque estable (core + guia + taste +
        # docs/digests) se cachea 5 min en la API → re-llamadas por seed/dials con
        # los mismos toggles pagan ese bloque a ~0.1x. Sin widget para no descuadrar
        # widgets_values de workflows guardados.
        system_param = [
            {"type": "text", "text": stable_block, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": variable_block},
        ]

        # creativity (0-1) -> temperature
        temperature = max(0.0, min(1.0, float(creativity)))

        try:
            api_kwargs = dict(
                model=claude_model.strip() or "claude-sonnet-4-6",
                # 4096 base: FLUX.1 legacy emite ~2x (prompt preview + clip_l + t5xxl +
                # negative) y con 2048 el JSON llegaba truncado -> 'Unterminated string'
                # -> fallback raw como prompt (bug verificado 5 jul 2026, bot.out.log).
                max_tokens=(8192 if int(variants) >= 4 else 4096),
                temperature=temperature,
                system=system_param,
                messages=[{"role": "user", "content": scene}],
            )
            message = self._create_with_fallback(api_kwargs)

            raw = self._extract_text(message)
            if not raw:
                raise RuntimeError("La API no devolvio contenido de texto")

            usage = getattr(message, "usage", None)
            if usage is not None:
                cache_w = getattr(usage, "cache_creation_input_tokens", 0) or 0
                cache_r = getattr(usage, "cache_read_input_tokens", 0) or 0
                print(f"[ClaudePromptGenerator] mode={mode} model={target_model} seed={seed} "
                      f"in:{usage.input_tokens} out:{usage.output_tokens} "
                      f"cache_w:{cache_w} cache_r:{cache_r}")
            print(f"[ClaudePromptGenerator] Raw: {repr(raw[:200])}")

            # Strip markdown fences si Claude las incluye
            if raw.startswith("```"):
                raw = raw.strip("`\n ")
                if raw.startswith("json"):
                    raw = raw[4:].strip()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[ClaudePromptGenerator] JSON parse error: {e}")
                return {
                    "ui": {"prompt": [raw], "razonamiento": ["Error al parsear JSON"]},
                    "result": (raw, "Error al parsear JSON", "", "", "")
                }

            prompt = (data.get("prompt") or "").strip()
            clip_l = (data.get("clip_l") or "").strip()
            t5xxl = (data.get("t5xxl") or "").strip()
            negative = (data.get("negative") or "").strip()
            razonamiento = (data.get("razonamiento") or "").strip()
            var_list = data.get("variants") or []
            if not isinstance(var_list, list):
                var_list = []

            # Fallbacks de coherencia
            if not prompt and clip_l:
                prompt = (clip_l + "\n\n" + t5xxl).strip()
            if not prompt and var_list:
                prompt = str(var_list[0]).strip()
            if not prompt:
                raise ValueError("Campo 'prompt' vacio en JSON")

            # Razonamiento enriquecido para inspeccion en el nodo
            extra_view = []
            if negative:
                extra_view.append("NEGATIVE:\n" + negative)
            if clip_l or t5xxl:
                extra_view.append("CLIP_L:\n" + clip_l + "\n\nT5XXL:\n" + t5xxl)
            if len(var_list) > 1:
                vv = "\n\n".join(f"[{i+1}] {str(v).strip()}" for i, v in enumerate(var_list))
                extra_view.append("VARIANTES:\n" + vv)
            razonamiento_full = razonamiento
            if extra_view:
                razonamiento_full = (razonamiento + "\n\n" + ("\n\n----\n\n".join(extra_view))).strip()

            return {
                "ui": {"prompt": [prompt], "razonamiento": [razonamiento_full]},
                "result": (prompt, razonamiento_full, clip_l, t5xxl, negative)
            }

        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"Connection error: {str(e)}")
        except anthropic.RateLimitError as e:
            raise RuntimeError(f"Rate limit: {str(e)}")
        except anthropic.APIError as e:
            raise RuntimeError(f"API Error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error inesperado: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "ClaudePromptGenerator": ClaudePromptGenerator
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClaudePromptGenerator": "Claude Prompt Generator (Rafa)"
}
