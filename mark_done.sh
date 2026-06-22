#!/bin/bash
# Crea retroactivamente los .done que faltan para runs ya completadas.
#
# Una run está completada si existe su molecules.csv (mismo criterio que los
# train_*.sh). El .done va JUNTO a la carpeta run_NN:
#
#   results/<cx>_<mut>/<ALG>/pop<POP>/run_NN/molecules.csv  ->  run_NN.done
#
# Es genérico: recorre todos los molecules.csv bajo la raíz, así que cubre los
# 5 algoritmos y todas las combinaciones de operadores de una sola pasada.
#
# Uso:
#   ./mark_done.sh             # usa ./results
#   ./mark_done.sh <dir>       # raíz de resultados alternativa
#   DRY_RUN=1 ./mark_done.sh   # solo muestra qué crearía, sin tocar nada

set -euo pipefail

ROOT="${1:-results}"

if [ ! -d "$ROOT" ]; then
    echo "ERROR: no existe el directorio '$ROOT'" >&2
    exit 1
fi

created=0
existing=0
total=0

while IFS= read -r -d '' csv; do
    total=$((total + 1))
    run_dir="$(dirname "$csv")"      # .../pop300/run_01
    done_file="${run_dir}.done"      # .../pop300/run_01.done

    if [ -f "$done_file" ]; then
        existing=$((existing + 1))
        continue
    fi

    if [ -n "${DRY_RUN:-}" ]; then
        echo "[CREARÍA] $done_file"
    else
        touch "$done_file"
        echo "[OK]      $done_file"
    fi
    created=$((created + 1))
done < <(find "$ROOT" -type f -name molecules.csv -print0)

echo "------------------------------------------------------"
echo "  runs completadas (molecules.csv):  $total"
echo "  .done ya existentes:               $existing"
if [ -n "${DRY_RUN:-}" ]; then
    echo "  .done que se crearían:             $created  (DRY_RUN: no se tocó nada)"
else
    echo "  .done creados ahora:               $created"
fi
