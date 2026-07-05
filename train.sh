#!/bin/bash
#SBATCH --job-name=MO_all
#SBATCH --output=logs/mo_%j.out
#SBATCH --error=logs/mo_%j.err
#SBATCH --partition=toko06          # <-- AJUSTA al nombre real de tu partición CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64       # el nodo entero (tienes un 2º nodo libre)
#SBATCH --time=12:00:00

# ══════════════════════════════════════════════════════════════════════════════
#  Experimentos multi-objetivo — TODO en un solo job, paralelizado en CPU.
#
#  Reemplaza los 5 train_*.sh. El trabajo es CPU/memory-bound (RDKit + decode),
#  la GPU apenas aportaba (~1.8x) y su cola es de 1-2 días: por eso corre en CPU.
#
#  Las 340 runs (4 algos GA × 4 operadores × 20  +  MOPSO × 20) son independientes
#  y se lanzan concurrentes con `xargs -P`. Cada proceso usa 1 hilo (OMP=1), así
#  PARALLEL controla la concurrencia real. Reanudable vía .done por run.
# ══════════════════════════════════════════════════════════════════════════════

source /etc/profile

# ─── Entorno ──────────────────────────────────────────────────────────────────
# python del env directo (NO `conda activate`: segfaultea por bytecode corrupto).
export PYTHONDONTWRITEBYTECODE=1     # /home NFS casi lleno: no escribir .pyc
export OMP_NUM_THREADS=1             # 1 hilo/proceso -> PARALLEL = concurrencia real
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""       # forzar CPU
PYTHON=/home/cperez/miniconda3/envs/pymoo_env/bin/python

# ─── Concurrencia ─────────────────────────────────────────────────────────────
# Nº de runs en paralelo. Mídelo con scaling_bench.py EN EL NODO y pon aquí el
# "codo" (donde el throughput deja de subir). El trabajo es memory-bound y satura
# ANTES del nº de cores: NO pongas 64 a ciegas.  Override: PARALLEL=32 sbatch ...
PARALLEL=${PARALLEL:-16}

POP=300
# N_RUNS por config. Override para smoke-test: N_RUNS=1 sbatch train.sh  (17 runs)
N_RUNS=${N_RUNS:-20}

# Algoritmos GA (con operadores):  "ALG  script"
GA_ALGS=(
    "NSGA2   experimento_nsga2.py"
    "NSGA3   experimento_nsga3.py"
    "MOEAD   experimento_moead.py"
    "AGEMOEA experimento_agemoea.py"
)
# Combinaciones de operadores: "crossover mutation" (sbx_pm es el combo base)
OPERATORS=("sbx pm" "sbx gauss" "pcx pm" "pcx gauss")


# ─── Worker: ejecuta UNA run (lo invoca xargs en paralelo) ────────────────────
run_one() {
    local ALG="$1" SCRIPT="$2" CX="$3" MUT="$4" RUN="$5"

    local RESULTS_DIR="results/${CX}_${MUT}"
    local ALG_DIR="${RESULTS_DIR}/${ALG}/pop${POP}"
    local RUN_NN; RUN_NN=$(printf '%02d' $((RUN + 1)))
    local DONE_FILE="${ALG_DIR}/run_${RUN_NN}.done"
    local RUN_OUT="${ALG_DIR}/run_${RUN_NN}/molecules.csv"
    local LABEL="${ALG}[${CX}+${MUT}]/pop${POP}/run_${RUN_NN}"

    mkdir -p "$ALG_DIR"

    if [ -f "$DONE_FILE" ]; then
        echo "[$LABEL] ya completado, salto"
        return
    fi

    echo "[$LABEL] lanzando..."
    if [ "$ALG" = "MOPSO" ]; then
        "$PYTHON" "$SCRIPT" --pop_size "$POP" --run_id "$RUN"
    else
        "$PYTHON" "$SCRIPT" --pop_size "$POP" --run_id "$RUN" \
            --crossover "$CX" --mutation "$MUT"
    fi

    # .done atado a la existencia real del output, NO al exit code: estos nodos
    # pueden segfaultear en el teardown DESPUÉS de guardar los resultados.
    if [ -f "$RUN_OUT" ]; then
        touch "$DONE_FILE"
    else
        echo "[$LABEL] FALLÓ (sin molecules.csv), se reintentará en la próxima corrida"
    fi
}
export -f run_one
export PYTHON POP


# ─── Genera la lista de las 340 tareas (una por línea: ALG SCRIPT CX MUT RUN) ──
build_tasks() {
    local entry ALG SCRIPT OP CX MUT RUN
    for entry in "${GA_ALGS[@]}"; do
        read -r ALG SCRIPT <<< "$entry"
        for OP in "${OPERATORS[@]}"; do
            read -r CX MUT <<< "$OP"
            for RUN in $(seq 0 $((N_RUNS - 1))); do
                echo "$ALG $SCRIPT $CX $MUT $RUN"
            done
        done
    done
    # MOPSO: sin operadores GA, vive en el combo base sbx_pm
    for RUN in $(seq 0 $((N_RUNS - 1))); do
        echo "MOPSO experimento_mopso.py sbx pm $RUN"
    done
}

TOTAL=$(build_tasks | wc -l)
echo "======================================================"
echo "  Experimentos MO — QED(↑) SA(↓) Lipinski(↑)"
echo "  Total de runs      : $TOTAL"
echo "  Concurrencia (-P)  : $PARALLEL"
echo "  Nodo               : $(hostname)   cores: $(nproc)"
echo "======================================================"

# ─── Lanza todo en paralelo ───────────────────────────────────────────────────
# -L 1: una línea (5 campos) por invocación.  El '_' es $0 placeholder de bash -c.
build_tasks | xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _

# ─── Resúmenes (rápidos, seriales) al terminar TODAS las runs ─────────────────
echo "======================================================"
echo "  Generando resúmenes..."
echo "======================================================"
for entry in "${GA_ALGS[@]}"; do
    read -r ALG SCRIPT <<< "$entry"
    for OP in "${OPERATORS[@]}"; do
        read -r CX MUT <<< "$OP"
        "$PYTHON" "$SCRIPT" --pop_size "$POP" --crossover "$CX" --mutation "$MUT" --generate_summary
    done
done
"$PYTHON" experimento_mopso.py --pop_size "$POP" --generate_summary

echo "======================================================"
echo "  ¡COMPLETADO!  ($TOTAL runs, concurrencia $PARALLEL)"
echo "======================================================"
