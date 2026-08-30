"""
Orquestador del grid de sensibilidad de hiperparámetros.

Corre las 513 configuraciones × N_RUNS semillas en paralelo, con presupuesto fijo
de 100.000 evaluaciones cada una: 108 por cada GA (reparto pob×gen × operadores ×
probabilidades) más 81 de CMOPSO.  Reanudable: una run cuenta como completa si
existe su molecules.csv.

Cada perilla barrida queda en el path (results/<ALG>/<slug>/run_k) y como columna
de metrics.csv; al final se consolida en results/all_metrics.csv.

    python run_experiments.py
"""

import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from experimento import ALGS_GA
from utils_mo import ga_run_dir, cmopso_run_dir, consolidate_all, FSP3_MIN

ROOT   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable   # el python del entorno actual
# Los cinco algoritmos entran por el mismo script, con --alg.
EXPERIMENTO = "experimento.py"

# ─── Espacio de hiperparámetros ───────────────────────────────────────────────
POP_GEN   = [(100, 1000), (200, 500), (400, 250)]   # los 3 = 100.000 evaluaciones

# GA: probabilidad de cruce, de mutación por-gen y combos de operadores.
CX_PROBS  = [0.7, 0.9, 1.0]
MUT_PROBS = [0.004, 0.012, 0.031]
OPERATORS = [("sbx", "pm"), ("sbx", "gauss"), ("pcx", "pm"), ("pcx", "gauss")]

# CMOPSO: sus propias perillas.  La mutación se barre con los mismos valores que
# los GA para que sea comparable.
ELITE_SIZES = [5, 10, 25]
VEL_RATES   = [0.1, 0.2, 0.35]

GA_ALGS = ALGS_GA


# ─── Definición de tareas ─────────────────────────────────────────────────────

def build_tasks(n_runs):
    """Lista de tareas del grid."""
    tasks = []
    for alg in GA_ALGS:
        for pop, gen in POP_GEN:
            for cx, mut in OPERATORS:
                for cxp in CX_PROBS:
                    for mutp in MUT_PROBS:
                        for run in range(n_runs):
                            tasks.append(dict(kind="ga", alg=alg,
                                              pop=pop, gen=gen, cx=cx, mut=mut,
                                              cxp=cxp, mutp=mutp, run=run))
    for pop, gen in POP_GEN:
        for es in ELITE_SIZES:
            for mutp in MUT_PROBS:
                for vel in VEL_RATES:
                    for run in range(n_runs):
                        tasks.append(dict(kind="cmopso", alg="CMOPSO",
                                          pop=pop, gen=gen, es=es, mutp=mutp,
                                          vel=vel, run=run))
    return tasks


def run_dir_of(t):
    """Path de la run, el mismo que arma experimento.py."""
    if t['kind'] == 'cmopso':
        return cmopso_run_dir(t['pop'], t['gen'], t['es'], t['mutp'], t['vel'], t['run'])
    return ga_run_dir(t['alg'], t['cx'], t['mut'], t['cxp'], t['mutp'],
                      t['pop'], t['gen'], t['run'])


def is_done(t):
    """Una run está completa si existe su molecules.csv."""
    return os.path.exists(os.path.join(run_dir_of(t), "molecules.csv"))


def label(t):
    cfg = f"pop{t['pop']}xgen{t['gen']}/run_{t['run'] + 1:02d}"
    if t['kind'] == 'cmopso':
        return f"CMOPSO[e{t['es']:g}_mut{t['mutp']:g}_vel{t['vel']:g}]/{cfg}"
    return f"{t['alg']}[{t['cx']}{t['cxp']:g}+{t['mut']}{t['mutp']:g}]/{cfg}"


# ─── Ejecución de una run (subproceso aislado) ────────────────────────────────

def run_one(t, device, threads):
    """Lanza el script del experimento en un subproceso. Devuelve (t, ok, dt, stdout)."""
    env = dict(os.environ)
    # Limitar los hilos evita sobresuscripción entre workers.
    env["OMP_NUM_THREADS"]      = str(threads)
    env["MKL_NUM_THREADS"]      = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["NUMEXPR_NUM_THREADS"]  = str(threads)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""   # evita inicializar la GPU

    cmd = [PYTHON, os.path.join(ROOT, EXPERIMENTO), "--alg", t['alg'],
           "--pop_size", str(t['pop']), "--n_gen", str(t['gen']),
           "--run_id", str(t['run']), "--device", device]
    if t['kind'] == 'cmopso':
        cmd += ["--elite_size", str(t['es']), "--mut_prob", str(t['mutp']),
                "--vel_rate", str(t['vel'])]
    else:
        cmd += ["--crossover", t['cx'], "--mutation", t['mut'],
                "--cx_prob", str(t['cxp']), "--mut_prob", str(t['mutp'])]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return t, is_done(t), time.time() - t0, proc.stdout


# ─── Utilidades de progreso ───────────────────────────────────────────────────

def human_time(s):
    s = int(s)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, _ = divmod(s, 60)
    if d:  return f"{d}d {h}h {m}m"
    if h:  return f"{h}h {m}m"
    return f"{m}m"


def resolve_device(name):
    """'auto' → 'cuda' si hay GPU, si no 'cpu'."""
    if name != "auto":
        return name
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def default_parallel(device):
    """Runs concurrentes por defecto: en GPU manda la VRAM, en CPU los núcleos."""
    ncpu = os.cpu_count() or 2
    return 4 if device == "cuda" else max(1, min(ncpu - 1, 11))


def prewarm_caches():
    """Construye el cache de SMILES de MOSES una sola vez, para que no lo hagan
    los N workers en paralelo."""
    code = ("import utils_mo; "
            "print('  MOSES:', len(utils_mo._load_moses_train_smiles()), 'SMILES')")
    print(f"[{time.strftime('%F %T')}] preparando cache de MOSES...", flush=True)
    subprocess.run([PYTHON, "-c", code], cwd=ROOT)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Orquestador del análisis de sensibilidad de hiperparámetros MO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--n-runs",  type=int, default=20, help="Semillas por configuración.")
    ap.add_argument("--device",  choices=["auto", "cpu", "cuda"], default="auto",
                    help="Dispositivo para el VAE (auto → GPU si hay CUDA).")
    ap.add_argument("-p", "--parallel", type=int, default=None,
                    help="Runs concurrentes (default: 4 en GPU, ~núcleos-1 en CPU).")
    ap.add_argument("--summary-only", action="store_true",
                    help="No corre experimentos; solo consolida results/all_metrics.csv.")
    args = ap.parse_args()

    # Line-buffering: mantiene el orden con la salida de los subprocesos.
    sys.stdout.reconfigure(line_buffering=True)

    if args.summary_only:
        consolidate_all()
        return

    device   = resolve_device(args.device)
    parallel = args.parallel or default_parallel(device)
    threads  = max(1, (os.cpu_count() or 1) // parallel)

    tasks   = build_tasks(args.n_runs)
    pending = [t for t in tasks if not is_done(t)]
    total, done0 = len(tasks), len(tasks) - len(pending)
    n_ga    = len(GA_ALGS) * len(POP_GEN) * len(OPERATORS) * len(CX_PROBS) * len(MUT_PROBS)
    n_cmopso = len(POP_GEN) * len(ELITE_SIZES) * len(MUT_PROBS) * len(VEL_RATES)

    print("=" * 54)
    print(f"  Sensibilidad de hiperparámetros — QED(↑) SA(↓) | Fsp3 ≥ {FSP3_MIN}")
    print(f"  Máquina        : {os.uname().nodename}  ({os.cpu_count()} núcleos)")
    print(f"  Dispositivo    : {device}")
    print(f"  Concurrencia   : {parallel} runs  ({threads} hilos/run)")
    print(f"  Configs        : {n_ga} GA + {n_cmopso} CMOPSO = {n_ga + n_cmopso}")
    print(f"  Total de runs  : {total}   (ya hechas: {done0}, pendientes: {len(pending)})")
    print("=" * 54)

    if not pending:
        print("Todas las runs ya estaban completas.")
        consolidate_all()
        return

    prewarm_caches()

    t_start = time.time()
    completed = 0
    failed = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(run_one, t, device, threads): t for t in pending}
        for fut in as_completed(futures):
            t, ok, dt, out = fut.result()
            completed += 1
            elapsed = time.time() - t_start
            rate = completed / elapsed * 60 if elapsed > 0 else 0
            eta  = human_time((len(pending) - completed) / rate * 60) if rate > 0 else "…"
            status = "ok" if ok else "FALLÓ"
            print(f"[{time.strftime('%T')}] {done0 + completed}/{total} | "
                  f"{status:5s} {label(t)} ({dt:.0f}s) | "
                  f"{rate:.1f} runs/min | ETA {eta}", flush=True)
            if not ok:
                failed.append(t)
                tail = "\n".join(out.strip().splitlines()[-8:])
                print(f"  └─ sin molecules.csv. Últimas líneas:\n{tail}", flush=True)

    print("=" * 54)
    print(f"  FIN: {total - len(failed)}/{total} completas en {human_time(time.time() - t_start)}")
    if failed:
        print(f"  {len(failed)} fallaron esta corrida → reejecuta para reintentarlas:")
        for t in failed:
            print(f"    - {label(t)}")
    else:
        print("  Todas las runs completas.")
    print("=" * 54)

    consolidate_all()


if __name__ == "__main__":
    main()
