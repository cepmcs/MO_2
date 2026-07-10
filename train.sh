#!/bin/bash
#SBATCH --job-name=MO_all
#SBATCH --output=logs/mo_%j.out
#SBATCH --error=logs/mo_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu1          # nodo GPU (quítalo para que SLURM elija en la partición)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # el nodo entero (16 cores)
#SBATCH --exclusive              # reserva el nodo completo: nadie más corre en paralelo
#SBATCH --time=12:00:00          # límite de la partición GPU (12 h); relanzar para continuar

# ══════════════════════════════════════════════════════════════════════════════
#  Wrapper SLURM: solo pide GPU + entorno y delega TODO en run_experiments.py.
#  Ese script arma el grid (4 GA × 4 operadores × N_RUNS + MOPSO × N_RUNS), corre
#  en paralelo, reanuda por molecules.csv, muestra ETA y genera los resúmenes.
#  Nada de lógica de experimentos vive aquí: el .py es la única fuente de verdad.
# ══════════════════════════════════════════════════════════════════════════════

source /etc/profile
# module load cuda        # descomenta si tu torch NO trae runtime CUDA propio

# ─── Entorno ──────────────────────────────────────────────────────────────────
export PYTHONDONTWRITEBYTECODE=1     # /home NFS casi lleno: no escribir .pyc
# Sin GRES: el nodo gpu1 se reserva entero (--exclusive) y torch ve la GPU directo.
# No fijar CUDA_VISIBLE_DEVICES aquí (dejaría a torch sin GPU).

# python del env directo (NO `conda activate`). Override: PYTHON=/otra/ruta sbatch train.sh
PYTHON=${PYTHON:-/home/cperez/miniconda3/envs/pymoo_env/bin/python}

# ─── Parámetros (todos overrideables por variable de entorno al hacer sbatch) ──
DEVICE=${DEVICE:-cuda}                              # cuda | auto | cpu
PARALLEL=${PARALLEL:-8}                             # cuántas runs corren AL MISMO TIEMPO
N_RUNS=${N_RUNS:-20}                                # smoke test: N_RUNS=1 sbatch train.sh

mkdir -p logs

# ─── Preflight: si pedimos GPU, verificar que torch REALMENTE la ve ───────────
# Falla en segundos en vez de desperdiciar el job (o correr 3 días en CPU sin querer).
if [ "$DEVICE" = "cuda" ]; then
    "$PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || { echo "ERROR: torch no ve CUDA en $(hostname). Revisa --gres, el env ($PYTHON) o 'module load cuda'." >&2; exit 1; }
    echo "[$(date '+%F %T')] CUDA OK: $("$PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

echo "======================================================"
echo "  Sensibilidad de hiperparámetros MO — SA(↓) d(ALOGP)(↑) d(HBD)(↑)"
echo "  Nodo         : $(hostname)   cores: $(nproc)   device: $DEVICE"
echo "  Concurrencia : $PARALLEL runs   n_runs: $N_RUNS"
echo "======================================================"

# ─── Todo lo demás (grid, paralelismo, reanudación, ETA, consolidación) vive aquí ──
exec "$PYTHON" run_experiments.py \
    --device "$DEVICE" \
    --parallel "$PARALLEL" \
    --n-runs "$N_RUNS"
