#!/bin/bash
#SBATCH --job-name=MO_all
#SBATCH --output=logs/mo_%j.out
#SBATCH --error=logs/mo_%j.err
#SBATCH --partition=XL           # <-- partición CPU; ajústala a tu clúster
#SBATCH --nodelist=toko06        # <-- nodo concreto; ajústalo (o quítalo para que SLURM elija)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64       # el nodo entero
#SBATCH --exclusive
#SBATCH --time=3-00:00:00        # máximo de la partición (3 días)

# ══════════════════════════════════════════════════════════════════════════════
#  Experimentos multi-objetivo — TODO en un solo job, paralelizado en CPU.
#
#  El trabajo es CPU/memory-bound (RDKit + decode), la GPU apenas aportaba
#  (~1.8x) y su cola es de 1-2 días: por eso corre en CPU.
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
# Nº de runs en paralelo. Con runs REALES (n_gen=500, eval_log de ~150k filas por
# run) a P=32 el nodo terminó OOMeando en cascada (Killed en el .err) cuando varias
# runs llegaban juntas al pico de memoria del post-procesamiento. A 16 no se vio ese
# problema. Override: PARALLEL=24 sbatch ...
PARALLEL=${PARALLEL:-16}

# Progreso: cada PROGRESS_EVERY s se escribe una línea "hechas/total + ETA" en un
# .out aparte (logs/progress_<jobid>.out). Es solo un contador de .done: no toca las
# runs ni las frena. Override para smoke-tests: PROGRESS_EVERY=10 sbatch ...
PROGRESS_EVERY=${PROGRESS_EVERY:-60}
PROGRESS_OUT="logs/progress_${SLURM_JOB_ID:-local}.out"

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


# ─── Monitor de progreso (contador de .done → hechas/total + ETA) ─────────────
count_done() { find results -name 'run_*.done' 2>/dev/null | wc -l; }

human_time() {   # segundos → "Xd Yh Zm"
    local s=$1 d h m
    d=$(( s / 86400 )); s=$(( s % 86400 ))
    h=$(( s / 3600  )); s=$(( s % 3600  ))
    m=$(( s / 60 ))
    if   [ "$d" -gt 0 ]; then printf '%dd %dh %dm' "$d" "$h" "$m"
    elif [ "$h" -gt 0 ]; then printf '%dh %dm' "$h" "$m"
    else                      printf '%dm' "$m"
    fi
}

# Corre en background durante el xargs. ETA = restantes × (tiempo/hecha), medido
# SOLO sobre lo completado en ESTA corrida (ignora los .done de corridas previas).
progress_monitor() {
    local total="$1" start_done="$2" interval="$3"
    local t0 now done_now completed remaining elapsed eta
    t0=$(date +%s)
    while true; do
        sleep "$interval"
        done_now=$(count_done)
        now=$(date +%s)
        completed=$(( done_now - start_done ))
        remaining=$(( total - done_now ))
        elapsed=$(( now - t0 ))
        if [ "$completed" -le 0 ]; then
            eta="estimando..."
        else
            eta=$(human_time $(( remaining * elapsed / completed )))
        fi
        printf '[%s] %d/%d hechas (%d en esta corrida) | restante estimado: %s\n' \
            "$(date '+%F %T')" "$done_now" "$total" "$completed" "$eta" >> "$PROGRESS_OUT"
    done
}


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
# Monitor de progreso en background (ver logs/progress_<jobid>.out).
mkdir -p logs
START_DONE=$(count_done)
echo "[$(date '+%F %T')] arranque: $START_DONE/$TOTAL ya hechas de corridas previas" > "$PROGRESS_OUT"
progress_monitor "$TOTAL" "$START_DONE" "$PROGRESS_EVERY" &
MON_PID=$!

# -L 1: una línea (5 campos) por invocación.  El '_' es $0 placeholder de bash -c.
build_tasks | xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _

# Apaga el monitor y escribe la línea final.
kill "$MON_PID" 2>/dev/null; wait "$MON_PID" 2>/dev/null
echo "[$(date '+%F %T')] FIN: $(count_done)/$TOTAL hechas" >> "$PROGRESS_OUT"

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
