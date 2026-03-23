rafa-node-resize
Custom nodes para ComfyUI. Actualmente incluye dos herramientas:

Nodo 1 — Tamaño del nodo (menú contextual)
Añade "Tamaño del nodo..." al menú contextual (click derecho) de cualquier nodo.
Permite fijar ancho y alto con valores numéricos exactos. Muestra el tamaño actual, el mínimo calculado por LiteGraph, y avisa si el valor introducido está por debajo del mínimo.
Uso
Click derecho sobre cualquier nodo → Tamaño del nodo...

Muestra ancho y alto actual
Muestra el tamaño mínimo calculado por LiteGraph para ese nodo
Aviso naranja si el valor introducido está por debajo del mínimo (LiteGraph lo ignoraría)
Al aplicar, clampea automáticamente al mínimo si el valor es menor
Enter para aplicar, Escape o clic fuera para cerrar

Notas

Solo nodos — no afecta a grupos ni reroutes
Requiere ComfyUI con soporte de extensiones JS (frontend ≥ v1.x)


Nodo 2 — Resolución Preset (Rafa)
Selector de resoluciones estándar para FLUX y WAN. Aparece en la categoría rafa del menú de nodos.
Outputs: width y height como INT, listos para conectar directamente a cualquier nodo que los necesite.
Presets disponibles
PresetVerticalHorizontalFLUX — Retrato (1024×1024)1024 × 10241024 × 1024FLUX — Cuerpo entero (832×1216)832 × 12161216 × 832WAN — 480p (480×832)480 × 832832 × 480WAN — 720p (720×1280)720 × 12801280 × 720WAN — Cuadrado (1024×1024)1024 × 10241024 × 1024
Uso

Añadir nodo Resolución Preset (Rafa) desde la categoría rafa
Elegir preset en el desplegable
Elegir orientación: Vertical (retrato) o Horizontal (paisaje)
Conectar width y height al nodo destino (Empty Latent Image, etc.)


Instalación
bashcd /ruta/a/ComfyUI/custom_nodes
git clone https://github.com/osuvense/rafa-node-resize.git
Reiniciar ComfyUI.
En RunPod
bashcd /workspace/ComfyUI/custom_nodes
git clone https://github.com/osuvense/rafa-node-resize.git

Notas generales

Probado en ComfyUI Desktop v0.16.4 (Mac) y ComfyUI en RunPod
El nodo de menú contextual es puro JS, no requiere Python
El nodo de resolución requiere reinicio completo de ComfyUI tras la instalación
