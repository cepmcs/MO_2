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
#  Wrapper SLURM: pide GPU + entorno, corre el grid y deja UN tar listo para bajar.
#
#    1. run_experiments.py       arma el grid (4 GA × 4 operadores × N_RUNS +
#                                CMOPSO), corre en paralelo, reanuda por
#                                molecules.csv.
#    2. python -m analisis etapa1  elige las 17 configuraciones ganadoras.
#    3. tar                      empaqueta esas 17 + all_metrics.csv + las
#                                figuras de la etapa 1.
#
#  La etapa 1 corre acá y no en el PC porque es la que decide cuáles de las 513
#  configuraciones sobreviven: elegir donde están los datos evita bajar el grid
#  entero y volver al cluster a buscar las ganadoras.
#
#  El grid tarda más de las 12 h de la partición: los jobs intermedios terminan
#  OK sin exportar, y el tar sale solo cuando el grid está entero.
#
#  Las baselines NO están acá: se corren y se analizan en el PC, aparte.
#
#  La lógica de los experimentos y del análisis vive en los .py; acá solo se los
#  encadena y se arma el tar.
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
EXPORT=${EXPORT:-1}                                 # EXPORT=0 sbatch train.sh → solo el grid

mkdir -p logs

# ─── Preflight: fallar en segundos, no después de 12 h ───────────────────────
# Si pedimos GPU, verificar que torch REALMENTE la ve (si no, el job se
# desperdicia o corre 3 días en CPU sin querer).
if [ "$DEVICE" = "cuda" ]; then
    "$PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || { echo "ERROR: torch no ve CUDA en $(hostname). Revisa --gres, el env ($PYTHON) o 'module load cuda'." >&2; exit 1; }
    echo "[$(date '+%F %T')] CUDA OK: $("$PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# La exportación importa la etapa 1 del análisis, que dibuja: si al env le falta
# matplotlib o scipy nos enteramos AHORA y no cuando el grid ya terminó.
if [ "$EXPORT" = "1" ]; then
    "$PYTHON" -c "import matplotlib, scipy, pandas" \
        || { echo "ERROR: al env ($PYTHON) le falta matplotlib/scipy/pandas, que usa la etapa 1 del análisis." >&2
             echo "       Instalalos, o corré con EXPORT=0 sbatch train.sh y exportá a mano después." >&2; exit 1; }
    echo "[$(date '+%F %T')] Dependencias del análisis OK."
fi

echo "======================================================"
echo "  Sensibilidad de hiperparámetros MO — QED(↑) SA(↓) | constraint Fsp3"
echo "  Nodo         : $(hostname)   cores: $(nproc)   device: $DEVICE"
echo "  Concurrencia : $PARALLEL runs   n_runs: $N_RUNS"
echo "  Exportar     : $EXPORT"
echo "======================================================"

# ─── 1. El grid (paralelismo, reanudación, ETA y consolidación viven ahí) ─────
"$PYTHON" run_experiments.py \
    --device "$DEVICE" \
    --parallel "$PARALLEL" \
    --n-runs "$N_RUNS"

# ─── 2. Etapa 1 + tar para bajar al PC ───────────────────────────────────────
[ "$EXPORT" = "1" ] || exit 0

SEL="plots/hiperparametros/selected_configs.csv"
TAR="MO2_analisis.tar"

# ¿Está completo el grid?  Una run cuenta como completa si escribió molecules.csv,
# el mismo criterio que usa run_experiments.py para reanudar.  El total lo calcula
# el propio grid: repetir el 513 acá sería una constante de más que puede divergir.
ESPERADAS=$("$PYTHON" -c "from run_experiments import build_tasks; print(len(build_tasks($N_RUNS)))")
HECHAS=$(find results -name molecules.csv | wc -l)
if [ "$HECHAS" -lt "$ESPERADAS" ]; then
    echo
    echo "Grid incompleto: $HECHAS/$ESPERADAS runs. No se exporta todavía."
    echo "Relanzá el job (es reanudable) y el tar sale solo al terminar."
    exit 0
fi
echo
echo "[$(date '+%F %T')] Grid completo: $HECHAS/$ESPERADAS runs. Exportando..."

# Etapa 1: elige las 17 ganadoras (la mejor de cada combo de operadores en los
# cuatro GA, más la mejor global de CMOPSO) y deja figuras, selected_configs.csv
# y la tabla LaTeX en plots/hiperparametros/.
"$PYTHON" -m analisis etapa1 --csv results/all_metrics.csv || exit 1

# selected_configs.csv → las rutas de esas 17 dentro de results/.  El %g de awk es
# el mismo formato C con que utils_mo._slug nombró las carpetas (1.0 → "1",
# 0.012 → "0.012").  LC_ALL=C no es opcional: con un locale de coma decimal awk
# leería 0.7 como 0 y armaría nombres rotos sin avisar.
CONFIGS=$(LC_ALL=C awk -F, '
    NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
    $(col["algorithm"]) == "CMOPSO" {
        printf "CMOPSO/pop%d_gen%d_e%g_mut%g_vel%g\n",
               $(col["pop_size"]), $(col["n_gen"]), $(col["elite_size"]),
               $(col["mut_prob"]), $(col["vel_rate"]); next }
    { printf "%s/%s_%s/cx%g_mut%g_pop%d_gen%d\n",
             $(col["algorithm"]), $(col["crossover"]), $(col["mutation"]),
             $(col["cx_prob"]), $(col["mut_prob"]),
             $(col["pop_size"]), $(col["n_gen"]) }
' "$SEL")

# Si algún nombre no existe, el formato de arriba se desincronizó de utils_mo:
# parar acá es mejor que bajar un tar al que le faltan configuraciones.
for c in $CONFIGS; do          # sin comillas a propósito: una ruta por línea, sin espacios
    [ -d "results/$c" ] || {
        echo "ERROR: la etapa 1 eligió 'results/$c', que no existe." >&2
        echo "       ¿Cambiaron ga_run_dir/cmopso_run_dir en utils_mo.py?" >&2
        exit 1; }
done

# El tar se extrae en la raíz del repo y cada cosa cae donde el análisis la busca
# (ver analisis/comun.py): grid/ solo con all_metrics.csv, que es lo único que se
# lee de él, y winners/ con las 17 runs completas.  --transform reescribe los
# nombres al vuelo, así que no hace falta copiar nada a un directorio intermedio.
# Sin comprimir: all_molecules.csv.gz ya viene en gzip y es el 98% del peso.
rm -f "$TAR"
tar cf "$TAR" -C results --transform 's|^|resultados/winners/|' $CONFIGS || exit 1
tar rf "$TAR" -C results --transform 's|^|resultados/grid/|' all_metrics.csv || exit 1
tar rf "$TAR" plots/hiperparametros || exit 1

echo
echo "======================================================"
echo "  ✅ $TAR  ($(du -h "$TAR" | cut -f1))"
echo "     $(echo "$CONFIGS" | wc -l) configuraciones + all_metrics.csv + etapa 1"
echo
echo "  En el PC, desde la raíz del repo:"
echo "     scp $(hostname):$(pwd)/$TAR ."
echo "     tar xf $TAR"
echo "     python -m analisis etapa2"
echo "======================================================"
