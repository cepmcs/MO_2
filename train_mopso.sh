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

# ─── Entorno conda ───────────────────────────────────────────────────────────
source /home/cperez/miniconda3/bin/activate

if ! conda info --envs | grep -q "pymoo_env"; then
    echo "Creando entorno conda pymoo_env..."
    conda create -n pymoo_env python=3.12 -y
    conda activate pymoo_env
    pip install torch pymoo rdkit pandas matplotlib numpy scipy
else
    conda activate pymoo_env
fi

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

        python experimento_mopso.py \
            --pop_size "$POP" \
            --run_id "$RUN" \
        && touch "$DONE_FILE"

    done

    echo "Generando resumen para ${ALG} pop${POP}..."
    python experimento_mopso.py --pop_size "$POP" --generate_summary

done

echo "======================================================"
echo "  ¡MOPSO COMPLETADO!"
for POP in "${POP_SIZES[@]}"; do
    echo "  pop${POP}: ${RESULTS_DIR}/${ALG}/pop${POP}/"
done
echo "======================================================"
