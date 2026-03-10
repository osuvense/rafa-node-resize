# rafa-node-resize

Custom node para ComfyUI que añade **"Tamaño del nodo..."** al menú contextual (click derecho) de cualquier nodo.

Permite fijar ancho y alto con valores numéricos exactos. Muestra el tamaño actual, el mínimo calculado por LiteGraph, y avisa si el valor introducido está por debajo del mínimo.

## Instalación

```bash
cd /ruta/a/ComfyUI/custom_nodes
git clone https://github.com/osuvense/rafa-node-resize.git
```

Reiniciar ComfyUI.

### En RunPod

```bash
cd /workspace/ComfyUI/custom_nodes
git clone https://github.com/osuvense/rafa-node-resize.git
```

## Uso

Click derecho sobre cualquier nodo → **Tamaño del nodo...**

- Muestra ancho y alto actual
- Muestra el tamaño mínimo calculado por LiteGraph para ese nodo
- Aviso naranja si el valor introducido está por debajo del mínimo (LiteGraph lo ignoraría)
- Al aplicar, clampea automáticamente al mínimo si el valor es menor
- Enter para aplicar, Escape o clic fuera para cerrar

## Notas

- Solo nodos — no afecta a grupos ni reroutes
- Requiere ComfyUI con soporte de extensiones JS (frontend ≥ v1.x)
- Probado en ComfyUI Desktop v0.16.4 (Mac) y ComfyUI en RunPod
