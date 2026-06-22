#!/bin/bash
#SBATCH --job-name=NSGA3_MO
#SBATCH --output=salida_nsga3.out
#SBATCH --error=error_nsga3.err
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

# ─── Configuración NSGA-III ───────────────────────────────────────────────────
ALG="NSGA3"
SCRIPT="experimento_nsga3.py"
POP_SIZES=(300)
N_RUNS=20

# Combinaciones de operadores: "crossover mutation"
# Cada combo escribe en results/<cx>_<mut>/ (sbx_pm es el base)
OPERATORS=(
    "sbx pm"
    "sbx gauss"
    "pcx pm"
    "pcx gauss"
)

echo "======================================================"
echo "  NSGA-III — Optimización Multi-Objetivo"
echo "  Objetivos: QED (↑), SA (↓), Lipinski (↑)"
echo "  Poblaciones: ${POP_SIZES[*]}"
echo "  Combinaciones de operadores: ${#OPERATORS[@]}"
echo "  Runs por config: $N_RUNS"
echo "  Total: $(( ${#POP_SIZES[@]} * ${#OPERATORS[@]} * N_RUNS )) ejecuciones"
echo "======================================================"

for OP in "${OPERATORS[@]}"; do
    read -r CX MUT <<< "$OP"

    # Directorio de resultados: un combo por carpeta (sbx_pm es el base)
    RESULTS_DIR="results/${CX}_${MUT}"

    echo "------------------------------------------------------"
    echo "  Operadores: crossover=${CX}  mutation=${MUT}  ->  ${RESULTS_DIR}"
    echo "------------------------------------------------------"

    for POP in "${POP_SIZES[@]}"; do

        ALG_DIR="${RESULTS_DIR}/${ALG}/pop${POP}"
        mkdir -p "$ALG_DIR"

        for RUN in $(seq 0 $((N_RUNS - 1))); do

            RUN_LABEL="${ALG}_${CX}_${MUT}_pop${POP}_run$((RUN + 1))"
            DONE_FILE="${ALG_DIR}/run_$(printf '%02d' $((RUN + 1))).done"

            if [ -f "$DONE_FILE" ]; then
                echo "[$RUN_LABEL] Ya completado. Saltando..."
                continue
            fi

            echo "[$RUN_LABEL] Lanzando..."

            "$PYTHON" "$SCRIPT" \
                --pop_size "$POP" \
                --run_id "$RUN" \
                --crossover "$CX" \
                --mutation "$MUT" \
            && touch "$DONE_FILE"

        done

        echo "Generando resumen para ${ALG} ${CX}+${MUT} pop${POP}..."
        "$PYTHON" "$SCRIPT" --pop_size "$POP" --crossover "$CX" --mutation "$MUT" --generate_summary

    done

done

echo "======================================================"
echo "  ¡NSGA-III COMPLETADO! (${#OPERATORS[@]} combinaciones de operadores)"
echo "======================================================"
