#!/bin/bash
#SBATCH --job-name=MO_all
#SBATCH --output=logs/mo_%j.out
#SBATCH --error=logs/mo_%j.err
#SBATCH --partition=XL           # <-- partición CPU; ajústala a tu clúster
#SBATCH --nodelist=toko01        # <-- nodo concreto; ajústalo (o quítalo para que SLURM elija)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64       # el nodo entero
#SBATCH --exclusive
#SBATCH --time=3-00:00:00        # máximo de la partición (3 días)

# ══════════════════════════════════════════════════════════════════════════════
#  Experimentos multi-objetivo — 340 runs (4 GA × 4 operadores × 20 + MOPSO × 20)
#  en un solo job, concurrentes con `xargs -P` (1 hilo/proceso, OMP=1). Reanudable
#  por .done. Corre en CPU: el trabajo es RDKit + decode, no GPU-bound.
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
# Nº de runs en paralelo (1 hilo/proceso -> P = concurrencia real). El trabajo
# satura por ancho de banda de memoria antes que por nº de cores: hay un "codo"
# pasado el cual sumar procesos no acelera. Calibra P empíricamente en el nodo real
# (no P = nº de cores a ciegas). Override: PARALLEL=24 sbatch ...
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

# Corre en background durante el xargs. ETA = restantes × (tiempo/hecha) y throughput
# (runs/min) medidos SOLO sobre lo completado en ESTA corrida (ignora los .done de
# corridas previas). El rate se calcula en entero *100 para 2 decimales sin float/awk.
progress_monitor() {
    local total="$1" start_done="$2" interval="$3"
    local t0 now done_now completed remaining elapsed eta rate rate_x100
    t0=$(date +%s)
    while true; do
        sleep "$interval"
        done_now=$(count_done)
        now=$(date +%s)
        completed=$(( done_now - start_done ))
        remaining=$(( total - done_now ))
        elapsed=$(( now - t0 ))
        if [ "$completed" -le 0 ] || [ "$elapsed" -le 0 ]; then
            eta="estimando..."; rate="0.00"
        else
            eta=$(human_time $(( remaining * elapsed / completed )))
            rate_x100=$(( completed * 6000 / elapsed ))
            rate=$(printf '%d.%02d' $(( rate_x100 / 100 )) $(( rate_x100 % 100 )))
        fi
        printf '[%s] %d/%d hechas | %d pendientes | %d en esta corrida | %s runs/min | ETA: %s\n' \
            "$(date '+%F %T')" "$done_now" "$total" "$remaining" "$completed" "$rate" "$eta" >> "$PROGRESS_OUT"
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

# ─── Pre-serializa MOSES train SMILES (una sola vez, serial) ──────────────────
# Cada run necesita los SMILES de train de MOSES al arrancar. Parsear el CSV de
# 1.9M filas en las N runs a la vez producía un pico de RAM simultáneo que disparaba
# el OOM. Esto lo construye UNA vez (pickle gzip en data/) de forma serial; luego
# cada run solo lo lee (barato). Idempotente: si ya existe y moses.csv no cambió, no
# reparsea nada.
echo "[$(date '+%F %T')] pre-serializando MOSES train SMILES (una vez)..."
"$PYTHON" -c "import utils_mo; print('  cache lista:', len(utils_mo._load_moses_train_smiles()), 'SMILES')" \
    || echo "  WARN: falló la pre-serialización; cada run parseará el CSV (más RAM al arranque)"

# ─── Lanza todo en paralelo ───────────────────────────────────────────────────
# Monitor de progreso en background (ver logs/progress_<jobid>.out).
mkdir -p logs
START_DONE=$(count_done)
echo "[$(date '+%F %T')] arranque: $START_DONE/$TOTAL ya hechas de corridas previas" > "$PROGRESS_OUT"
progress_monitor "$TOTAL" "$START_DONE" "$PROGRESS_EVERY" &
MON_PID=$!

# -L 1: una línea (5 campos) por invocación.  El '_' es $0 placeholder de bash -c.
build_tasks | xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _

# Apaga el monitor y escribe el resumen final (a stdout y al progress_*.out). Un
# .done = run completa, así que pendientes = TOTAL - hechas = runs que fallaron esta
# corrida (sin molecules.csv) y se reintentan relanzando el job.
kill "$MON_PID" 2>/dev/null; wait "$MON_PID" 2>/dev/null
FINAL_DONE=$(count_done)
NEW_DONE=$(( FINAL_DONE - START_DONE ))
FAILED=$(( TOTAL - FINAL_DONE ))
{
    echo "[$(date '+%F %T')] FIN: $FINAL_DONE/$TOTAL hechas ($NEW_DONE nuevas esta corrida)"
    if [ "$FAILED" -gt 0 ]; then
        echo "  $FAILED runs sin .done (fallaron esta corrida) -> relanza 'sbatch train.sh' para reintentarlas"
    else
        echo "  todas las runs completas"
    fi
} | tee -a "$PROGRESS_OUT"

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
