#!/bin/bash
#SBATCH --job-name=MO_sweep
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --partition=XL  
#SBATCH --nodelist=toko06        # misma partición CPU que train.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64        # nodo entero: para poder medir hasta N=64
#SBATCH --time=00:30:00

# ══════════════════════════════════════════════════════════════════════════════
#  Mide la curva de escalado EN EL NODO para elegir PARALLEL de train.sh.
#  Corre: sbatch sweep.sh   ->   luego lee logs/sweep_<jobid>.out
#  La tabla marca el "codo" (dónde el throughput deja de subir) = tu -P.
# ══════════════════════════════════════════════════════════════════════════════

source /etc/profile
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/cperez/miniconda3/envs/pymoo_env/bin/python

# scaling_bench ya fuerza CPU (CUDA_VISIBLE_DEVICES="") y 1 hilo por worker.
# Lanza N copias concurrentes (subprocesos hijos, igual que el xargs -P de train.sh).
"$PYTHON" scaling_bench.py sweep 300 20 "${TMPDIR:-/tmp}/mo_sweep_$SLURM_JOB_ID" 8,16,32,48,64
