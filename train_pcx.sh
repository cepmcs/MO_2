#!/bin/bash
#SBATCH --job-name=MO_pcx
#SBATCH --output=logs/mo_pcx_%j.out
#SBATCH --error=logs/mo_pcx_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu2          # pcx en gpu2, sbx en gpu1 (train_sbx.sh): corren a la vez
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # el nodo entero (16 cores)
#SBATCH --exclusive              # reserva el nodo completo: nadie más corre en paralelo
#SBATCH --time=12:00:00          # límite de la partición GPU (12 h); relanzar para continuar

source /etc/profile
# module load cuda        # descomenta si tu torch NO trae runtime CUDA propio

# ─── Entorno ──────────────────────────────────────────────────────────────────
export PYTHONDONTWRITEBYTECODE=1     # /home NFS casi lleno: no escribir .pyc
# No fijar CUDA_VISIBLE_DEVICES acá: dejaría a torch sin GPU.

# python del env directo (NO `conda activate`). Override: PYTHON=/otra/ruta sbatch train_pcx.sh
PYTHON=${PYTHON:-/home/cperez/miniconda3/envs/pymoo_env/bin/python}

# ─── Parámetros (todos overrideables por variable de entorno al hacer sbatch) ──
DEVICE=${DEVICE:-cuda}                              # cuda | auto | cpu
PARALLEL=${PARALLEL:-4}                             # cuántas runs corren AL MISMO TIEMPO
N_RUNS=${N_RUNS:-20}                                # smoke test: N_RUNS=1 sbatch train_pcx.sh
CRUCE=pcx                                           # este script solo corre este cruce

mkdir -p logs

# ─── Preflight: fallar en segundos, no después de 12 h ───────────────────────
if [ "$DEVICE" = "cuda" ]; then
    "$PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || { echo "ERROR: torch no ve CUDA en $(hostname). Revisa --gres, el env ($PYTHON) o 'module load cuda'." >&2; exit 1; }
    echo "[$(date '+%F %T')] CUDA OK: $("$PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

export UNIDOCK_PREFIX=${UNIDOCK_PREFIX:-${PYTHON%/envs/*}/envs/unidock}
"$PYTHON" docking.py --check >/dev/null \
    || { echo "ERROR: Uni-Dock no funciona en $(hostname) ($UNIDOCK_PREFIX). Env: conda create -p \$UNIDOCK_PREFIX -c conda-forge --override-channels python=3.12 unidock=1.2.0 meeko=0.8.0 rdkit" >&2; exit 1; }

echo "======================================================"
echo "  exp4 — QED(↑) SA(↓) Docking(↓) | constraint Fsp3"
echo "  Cruce        : $CRUCE"
echo "  Nodo         : $(hostname)   cores: $(nproc)   device: $DEVICE"
echo "  Concurrencia : $PARALLEL runs   n_runs: $N_RUNS"
echo "======================================================"

# ─── El grid, acotado a un solo cruce ─────────────────────────────────────────
# run_experiments.py toma los cruces de la lista global CRUCES al armar las tareas:
# acotarla acá reparte sbx y pcx en dos jobs sin tocar el orquestador.
# Al terminar, juntar las métricas de los dos jobs: python run_experiments.py --summary-only
"$PYTHON" -c "import run_experiments as r; r.CRUCES[:] = ['$CRUCE']; r.main()" \
    --device "$DEVICE" \
    --parallel "$PARALLEL" \
    --n-runs "$N_RUNS"
