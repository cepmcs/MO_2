#!/bin/bash
#SBATCH --job-name=MO_pilot
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu1          # nodo GPU (quítalo para que SLURM elija en la partición)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=12:00:00          # límite de la partición GPU (12 h)

# Piloto de paralelización con docking real: cuántas runs a la vez conviene correr en una GPU.
# La GPU la eliges al lanzarlo (CUDA_VISIBLE_DEVICES); todas las runs usan esa.
#   bash test.sh setup                                       crea el env de Uni-Dock si falta y lo prueba (necesita internet)
#   mkdir -p logs && CUDA_VISIBLE_DEVICES=1 sbatch test.sh   corre el piloto; la tabla queda en results/pilot/tabla.txt

source /etc/profile
export PYTHONDONTWRITEBYTECODE=1

PYTHON=${PYTHON:-/home/cperez/miniconda3/envs/pymoo_env/bin/python}
CONDA_BASE=${PYTHON%/envs/*}
export UNIDOCK_PREFIX=${UNIDOCK_PREFIX:-$CONDA_BASE/envs/unidock}
export N_GEN=${N_GEN:-100}                           # generaciones por run del piloto
ALG=${ALG:-NSGA2}
CONFIGS=${CONFIGS:-"1 2 4 8"}                        # runs a la vez
PILOT=${PILOT:-$PWD/results/pilot}

# ─── Uni-Dock: crear el env si falta y probarlo ───────────────────────────────
if [ ! -x "$UNIDOCK_PREFIX/bin/unidock" ]; then
    echo "[$(date '+%F %T')] creando el env de Uni-Dock en $UNIDOCK_PREFIX"
    "$CONDA_BASE/bin/conda" create -y -p "$UNIDOCK_PREFIX" -c conda-forge --override-channels \
        python=3.12 unidock=1.2.0 meeko=0.8.0 rdkit \
        || { echo "ERROR: no se pudo crear el env (¿sin internet en este nodo? corre 'bash test.sh setup' en el login)" >&2; exit 1; }
fi
"$PYTHON" docking.py --check \
    || { echo "ERROR: Uni-Dock no funciona en $(hostname) ($UNIDOCK_PREFIX). ¿El driver soporta CUDA 12.9?" >&2; exit 1; }
[ "$1" = "setup" ] && { echo "Uni-Dock OK"; exit 0; }

"$PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "ERROR: torch no ve CUDA en $(hostname)." >&2; exit 1; }
NCPU=$(nproc)
GPU=${CUDA_VISIBLE_DEVICES:-0}
GPU=${GPU%%,*}

# ─── Piloto ──────────────────────────────────────────────────────────────────
rm -rf "${PILOT:?}"
mkdir -p "$PILOT"
echo "runs,wall_s,fallos" > "$PILOT/resumen.csv"

for P in $CONFIGS; do
    D=$PILOT/p$P
    mkdir -p "$D"
    echo "[$(date '+%T')] $P runs a la vez, $N_GEN generaciones (GPU $GPU)"
    nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 5 > "$D/gpu.csv" &
    SAMPLER=$!
    t0=$(date +%s.%N)
    pids=()
    for ((i = 0; i < P; i++)); do
        RESULTS_DIR="$D" \
        OMP_NUM_THREADS=$((NCPU / P)) MKL_NUM_THREADS=$((NCPU / P)) OPENBLAS_NUM_THREADS=$((NCPU / P)) \
            "$PYTHON" experimento.py --alg "$ALG" --crossover sbx --run_id "$i" \
            --n_gen "$N_GEN" --device cuda > "$D/run$i.log" 2>&1 &
        pids+=($!)
    done
    fallos=0
    for p in "${pids[@]}"; do wait "$p" || fallos=$((fallos + 1)); done
    t1=$(date +%s.%N)
    kill "$SAMPLER" 2>/dev/null
    echo "$P,$(echo "$t1 - $t0" | bc),$fallos" >> "$PILOT/resumen.csv"
done

# ─── Tabla ───────────────────────────────────────────────────────────────────
"$PYTHON" - "$PILOT" <<'PY' | tee "$PILOT/tabla.txt"
import csv, glob, os, sys
import pandas as pd

pil, ngen = sys.argv[1], int(os.environ['N_GEN'])
filas = list(csv.DictReader(open(f'{pil}/resumen.csv')))
print(f"Piloto con {ngen} generaciones por run.  Las horas por run y del grid se extrapolan a 1000 "
      f"generaciones; las primeras generaciones dockean más moléculas nuevas que las últimas "
      f"(la caché crece), así que sobreestiman.\n")
print(f"{'runs':>4} {'pared min':>9} {'speedup':>8} {'h/run':>7} {'h grid':>7} "
      f"{'VRAM MiB':>9} {'GPU %':>6} {'fallos':>6} {'sin docking':>11}")
base, mejor = None, None
for r in filas:
    P, pared, fallos = int(r['runs']), float(r['wall_s']), int(r['fallos'])
    d = f'{pil}/p{P}'
    mets = [pd.read_csv(f) for f in glob.glob(f'{d}/**/metrics.csv', recursive=True)]
    por_run = pared / P
    base = base or por_run
    if fallos == 0 and (mejor is None or por_run < mejor[1]):
        mejor = (P, por_run)
    gpu = pd.read_csv(f'{d}/gpu.csv', header=None, names=['mem', 'util'])
    if mets:
        m = pd.concat(mets)
        h_run = m['time_sec'].mean() * 1000 / ngen / 3600
        sin_dock = int(m['n_dock_fail'].sum())
    else:
        h_run, sin_dock = float('nan'), -1
    h_grid = 120 * por_run * 1000 / ngen / 3600
    print(f"{P:>4} {pared / 60:>9.0f} {base / por_run:>7.2f}x {h_run:>7.1f} {h_grid:>7.0f} "
          f"{gpu['mem'].max():>9.0f} {gpu['util'].mean():>6.0f} {fallos:>6} {sin_dock:>11}")
if mejor:
    print(f"\nMás rápido: {mejor[0]} runs a la vez.")
PY
