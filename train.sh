#!/bin/bash
#SBATCH --job-name=MO_exp6
#SBATCH --output=logs/mo_exp6_%j.out
#SBATCH --error=logs/mo_exp6_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu1          # para pcx en el otro nodo: CRUCE=pcx sbatch --nodelist=gpu2 train.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # el nodo entero (16 cores)
#SBATCH --exclusive              # reserva el nodo completo: nadie más corre en paralelo
#SBATCH --time=12:00:00          # límite de la partición GPU (12 h)

# exp6: VAE original (sin fine-tuning), objetivos SA(↓) Docking(↓) pChEMBL(↑), sin QED,
# con las restricciones Fsp3 >= 0.3 y 1 <= logP <= 4.  Tres algoritmos, un cruce por job.
#   sbatch train.sh                              # cruce sbx (mutación por gen 0.1), en gpu1
#   CRUCE=pcx sbatch --nodelist=gpu2 train.sh    # cruce pcx (mutación por gen 0.012), en gpu2
#   N_RUNS=1 sbatch train.sh                     # prueba rápida: una run por configuración
#   PREFLIGHT_ONLY=1 bash train.sh               # en el nodo de login: revisa todo, sin correr nada
# Se puede lanzar varias veces seguidas: las de un mismo nodo hacen cola y, por el --exclusive, corren
# de a una, sin pisarse; cada una sigue con lo que falte.
# Todo se reanuda: las runs con molecules.csv se saltan, así que relanzar continúa donde quedó.
# (logs/ tiene que existir antes del sbatch: SLURM no lo crea.)

source /etc/profile
# module load cuda        # descomenta si tu torch NO trae runtime CUDA propio

# ─── Entorno ──────────────────────────────────────────────────────────────────
export PYTHONDONTWRITEBYTECODE=1     # /home NFS casi lleno: no escribir .pyc
# No fijar CUDA_VISIBLE_DEVICES acá: dejaría a torch sin GPU.

# python del env directo (NO `conda activate`). Override: PYTHON=/otra/ruta sbatch train.sh
PYTHON=${PYTHON:-/home/cperez/miniconda3/envs/pymoo_env/bin/python}

# ─── Parámetros (todos overrideables por variable de entorno al hacer sbatch) ──
DEVICE=${DEVICE:-cuda}                              # cuda | auto | cpu
PARALLEL=${PARALLEL:-4}                             # cuántas runs corren AL MISMO TIEMPO
N_RUNS=${N_RUNS:-5}                                 # semillas por configuración
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}                 # 1: solo revisar
CRUCE=${CRUCE:-sbx}                                 # sbx | pcx
case "$CRUCE" in
    sbx) MUT=0.1 ;;                                 # mutación por gen de cada cruce
    pcx) MUT=0.012 ;;
    *) echo "ERROR: CRUCE='$CRUCE' no es sbx ni pcx." >&2; exit 1 ;;
esac

EXP_NAME=exp6_vae_sinqed_logp
ARGS="--vae original --logp"

mkdir -p logs

echo "======================================================"
echo "  $EXP_NAME | cruce $CRUCE (mutación por gen: $MUT)"
echo "  Objetivos    : SA, Docking, pChEMBL (sin QED) | Fsp3 >= 0.3, 1 <= logP <= 4"
echo "  Nodo         : $(hostname)   cores: $(nproc)   device: $DEVICE"
echo "  Concurrencia : $PARALLEL runs   n_runs: $N_RUNS"
echo "======================================================"

# ─── Preflight: fallar en segundos, no después de 12 h (ni de la cola) ────────
# scikit-learn lo pide el modelo de pChEMBL.  Si falta, se instala acá.  --no-cache-dir: /home casi lleno.
if ! "$PYTHON" -c "import sklearn, joblib" 2>/dev/null; then
    echo "[$(date '+%F %T')] scikit-learn no está en $PYTHON: instalando (el modelo se guardó con la 1.9.1)..."
    "$PYTHON" -m pip install --quiet --no-cache-dir "scikit-learn==1.9.1" joblib \
        || "$PYTHON" -m pip install --quiet --no-cache-dir scikit-learn joblib \
        || { echo "ERROR: no se pudo instalar scikit-learn en $PYTHON (¿sin red en este nodo?). Instálalo a mano desde el nodo de login: $PYTHON -m pip install scikit-learn joblib" >&2; exit 1; }
fi
echo "[$(date '+%F %T')] scikit-learn $("$PYTHON" -c 'import sklearn; print(sklearn.__version__)')"

# Archivos que git ignora (*.pkl, *.pth) y hay que llevar a mano o con `git add -f`.
for f in data/random_forest_ic50.pkl SMILES_LSTM_2_256_300_lr1e4_b64.pth; do
    [ -f "$f" ] || { echo "ERROR: falta $f (git ignora *.pkl y *.pth: cópialo o súbelo con 'git add -f')." >&2; exit 1; }
done
"$PYTHON" -W ignore -c "from pchembl import PChEMBLPredictor; p = PChEMBLPredictor()(['CC(C)c1ccc(C(=O)Nc2cccc(C(=O)N3CCC3)c2)cc1']); print(f'[pChEMBL OK] ejemplo -> {p[0]:.2f} (esperado ~4.82)')" \
    || { echo "ERROR: el modelo de pChEMBL no carga en $PYTHON (versión de scikit-learn, rdkit o pandas)." >&2; exit 1; }

if [ "$PREFLIGHT_ONLY" = 1 ]; then
    echo "Preflight OK (sin GPU ni docking: eso se revisa dentro del job)."
    exit 0
fi

if [ "$DEVICE" = "cuda" ]; then
    "$PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || { echo "ERROR: torch no ve CUDA en $(hostname). Revisa --gres, el env ($PYTHON) o 'module load cuda'." >&2; exit 1; }
    echo "[$(date '+%F %T')] CUDA OK: $("$PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

export UNIDOCK_PREFIX=${UNIDOCK_PREFIX:-${PYTHON%/envs/*}/envs/unidock}
"$PYTHON" docking.py --check >/dev/null \
    || { echo "ERROR: Uni-Dock no funciona en $(hostname) ($UNIDOCK_PREFIX). Env: conda create -p \$UNIDOCK_PREFIX -c conda-forge --override-channels python=3.12 unidock=1.2.0 meeko=0.8.0 rdkit" >&2; exit 1; }

# ─── El experimento ───────────────────────────────────────────────────────────
# run_experiments.py toma los cruces de la lista global CRUCES al armar las tareas: acotarla
# (y MUT_PROBS) acá deja un solo cruce por job, con su mutación, sin tocar el orquestador.
echo "[$(date '+%F %T')] ===== $EXP_NAME ($CRUCE) ====="
# $ARGS va sin comillas a propósito: son varios argumentos.
"$PYTHON" -c "import run_experiments as r; r.CRUCES[:] = ['$CRUCE']; r.MUT_PROBS[:] = [$MUT]; r.main()" \
    --device "$DEVICE" \
    --parallel "$PARALLEL" \
    --n-runs "$N_RUNS" \
    --exp-name "$EXP_NAME" \
    $ARGS \
    || echo "AVISO: $EXP_NAME terminó con error (relanzar reintenta las runs que falten)."
echo "[$(date '+%F %T')] FIN ($CRUCE)"
