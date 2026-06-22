#!/bin/bash
#SBATCH --job-name=MOPSO_MO
#SBATCH --output=salida_mopso.out
#SBATCH --error=error_mopso.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=12:00:00

source /etc/profile

# ─── Entorno ──────────────────────────────────────────────────────────────────
# Usamos el python del env directamente. NO usar `conda activate`: segfaultea en
# los nodos por bytecode corrupto de conda (bad marshal data en conda-content-trust).
# PYTHONDONTWRITEBYTECODE: no escribir .pyc. El /home es NFS compartido y casi
# lleno; procesos concurrentes escribiendo bytecode corrompen los .pyc (bad marshal).
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/cperez/miniconda3/envs/pymoo_env/bin/python

ALG="MOPSO"
POP_SIZES=(300)
N_RUNS=20

# MOPSO no tiene operadores GA: vive en el combo base sbx_pm
RESULTS_DIR="results/sbx_pm"

echo "======================================================"
echo "  MOPSO — Optimización Multi-Objetivo"
echo "  Objetivos: QED (↑), SA (↓), Lipinski (↑)"
echo "  Poblaciones: ${POP_SIZES[*]}"
echo "  Runs por config: $N_RUNS"
echo "  Total: $(( ${#POP_SIZES[@]} * N_RUNS )) ejecuciones"
echo "======================================================"

for POP in "${POP_SIZES[@]}"; do

    ALG_DIR="${RESULTS_DIR}/${ALG}/pop${POP}"
    mkdir -p "$ALG_DIR"

    for RUN in $(seq 0 $((N_RUNS - 1))); do

        RUN_LABEL="${ALG}_pop${POP}_run$((RUN + 1))"
        DONE_FILE="${ALG_DIR}/run_$(printf '%02d' $((RUN + 1))).done"

        if [ -f "$DONE_FILE" ]; then
            echo "[$RUN_LABEL] Ya completado. Saltando..."
            continue
        fi

        echo "[$RUN_LABEL] Lanzando..."

        "$PYTHON" experimento_mopso.py \
            --pop_size "$POP" \
            --run_id "$RUN" \
        && touch "$DONE_FILE"

    done

    echo "Generando resumen para ${ALG} pop${POP}..."
    "$PYTHON" experimento_mopso.py --pop_size "$POP" --generate_summary

done

echo "======================================================"
echo "  ¡MOPSO COMPLETADO!"
for POP in "${POP_SIZES[@]}"; do
    echo "  pop${POP}: ${RESULTS_DIR}/${ALG}/pop${POP}/"
done
echo "======================================================"
