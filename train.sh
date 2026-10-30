#!/bin/bash
#SBATCH --job-name=MO_exp5
#SBATCH --output=logs/mo_exp5_%A_%a.out
#SBATCH --error=logs/mo_exp5_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-1              # tarea 0 = sbx, tarea 1 = pcx: dos jobs, cada uno en el nodo que SLURM tenga libre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # el nodo entero (16 cores)
#SBATCH --exclusive              # reserva el nodo completo: nadie más corre en paralelo
#SBATCH --time=12:00:00          # límite de la partición GPU (12 h); relanzar para continuar

# Los cinco experimentos de exp5, uno tras otro, con un solo cruce por job (sbx con mutación 0.1,
# pcx con 0.012, como en exp4).  Sin --nodelist: SLURM elige el nodo libre.
#   sbatch train.sh                          # las dos tareas del array (sbx y pcx)
#   sbatch --array=0 train.sh                # solo sbx
#   N_RUNS=1 SOLO=exp5_vaeft_pchembl_logp sbatch train.sh   # prueba rápida de un experimento
#   PREFLIGHT_ONLY=1 bash train.sh           # en el nodo de login: instala y revisa todo, sin correr nada
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
N_RUNS=${N_RUNS:-5}                                 # semillas por configuración (exp4 usó 20)
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}                 # 1: solo instalar y revisar
SOLO=${SOLO:-}                                      # texto que debe contener el nombre del experimento

# Cruce de este job: la tarea del array decide; fuera del array, CRUCE=sbx|pcx.
CRUCE=${CRUCE:-}
if [ -z "$CRUCE" ]; then
    case "${SLURM_ARRAY_TASK_ID:-}" in 0) CRUCE=sbx ;; 1) CRUCE=pcx ;; esac
fi
[ "$PREFLIGHT_ONLY" = 1 ] && CRUCE=${CRUCE:-sbx}
case "$CRUCE" in
    sbx) MUT=0.1 ;;                                 # mutación por gen de cada cruce
    pcx) MUT=0.012 ;;
    *) echo "ERROR: sin cruce. Usa sbatch (array 0-1) o CRUCE=sbx|pcx (con SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID)." >&2; exit 1 ;;
esac

# ─── Los experimentos: carpeta de resultados | argumentos de run_experiments.py ─
EXPERIMENTOS=(
    "exp5_vae_pchembl|--vae original --pchembl"
    "exp5_vae_pchembl_logp|--vae original --pchembl --logp"
    "exp5_vaeft|--vae mrp1"
    "exp5_vaeft_pchembl|--vae mrp1 --pchembl"
    "exp5_vaeft_pchembl_logp|--vae mrp1 --pchembl --logp"
)
SELECCION=()
for exp in "${EXPERIMENTOS[@]}"; do
    if [ -z "$SOLO" ] || [[ "${exp%%|*}" == *"$SOLO"* ]]; then SELECCION+=("$exp"); fi
done
[ ${#SELECCION[@]} -gt 0 ] || { echo "ERROR: SOLO='$SOLO' no coincide con ningún experimento." >&2; exit 1; }

mkdir -p logs

echo "======================================================"
echo "  exp5 — ${#SELECCION[@]} experimento(s) | cruce $CRUCE (mutación por gen: $MUT)"
for exp in "${SELECCION[@]}"; do echo "    ${exp%%|*}  ${exp#*|}"; done
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
for f in data/random_forest_ic50.pkl SMILES_LSTM_2_256_300_lr1e4_b64_mrp1.pth; do
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

# ─── Los experimentos, uno tras otro ──────────────────────────────────────────
# run_experiments.py toma los cruces de la lista global CRUCES al armar las tareas: acotarla
# (y MUT_PROBS) acá deja un solo cruce por job, con su mutación, sin tocar el orquestador.
for exp in "${SELECCION[@]}"; do
    nombre=${exp%%|*}
    args=${exp#*|}
    echo "[$(date '+%F %T')] ===== $nombre ($CRUCE) ====="
    # $args va sin comillas a propósito: son varios argumentos.
    "$PYTHON" -c "import run_experiments as r; r.CRUCES[:] = ['$CRUCE']; r.MUT_PROBS[:] = [$MUT]; r.main()" \
        --device "$DEVICE" \
        --parallel "$PARALLEL" \
        --n-runs "$N_RUNS" \
        --exp-name "$nombre" \
        $args \
        || echo "AVISO: $nombre terminó con error; sigo con el siguiente (relanzar reintenta las runs que falten)."
done
echo "[$(date '+%F %T')] FIN ($CRUCE)"
