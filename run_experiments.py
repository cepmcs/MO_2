"""
Orquestador local de los experimentos multi-objetivo.

Reemplaza el antiguo train.sh (SLURM). Corre el grid completo en esta máquina:
GPU para el decode del VAE + varios experimentos en paralelo (el cuello de botella
real es RDKit en CPU, no el decode). Reanudable: una run cuenta como completa si
existe su molecules.csv.

Grid: 4 GA (NSGA2, NSGA3, MOEAD, AGEMOEA) × 4 operadores × N_RUNS + MOPSO × N_RUNS.

Uso:
    python run_experiments.py                       # default: GPU, 4 en paralelo, 20 runs
    python run_experiments.py --parallel 5          # más workers (ojo con los 4 GB de VRAM)
    python run_experiments.py --device cpu -p 11    # máximo throughput en CPU (ignora la GPU)
    python run_experiments.py --n-runs 1            # smoke test (17 runs)
    python run_experiments.py --summary-only        # solo regenerar los resúmenes
"""

import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable   # el python del entorno actual (nada de rutas hardcodeadas)

# Algoritmos GA (usan operadores de crossover/mutation).
GA_ALGS = [
    ("NSGA2",   "experimento_nsga2.py"),
    ("NSGA3",   "experimento_nsga3.py"),
    ("MOEAD",   "experimento_moead.py"),
    ("AGEMOEA", "experimento_agemoea.py"),
]
# Combinaciones de operadores: (crossover, mutation). sbx_pm es el combo base.
OPERATORS = [("sbx", "pm"), ("sbx", "gauss"), ("pcx", "pm"), ("pcx", "gauss")]
BASELINE  = ("sbx", "pm")   # MOPSO no tiene operadores GA: vive en el combo base.


# ─── Definición de tareas ─────────────────────────────────────────────────────

def build_tasks(pop, n_runs):
    """Lista de tareas del grid. Cada tarea es un dict autocontenido."""
    tasks = []
    for alg, script in GA_ALGS:
        for cx, mut in OPERATORS:
            for run in range(n_runs):
                tasks.append(dict(alg=alg, script=script, cx=cx, mut=mut, run=run, pop=pop))
    for run in range(n_runs):
        tasks.append(dict(alg="MOPSO", script="experimento_mopso.py",
                          cx=BASELINE[0], mut=BASELINE[1], run=run, pop=pop))
    return tasks


def run_dir_of(t):
    return os.path.join(ROOT, "results", f"{t['cx']}_{t['mut']}",
                        t['alg'], f"pop{t['pop']}", f"run_{t['run'] + 1:02d}")


def is_done(t):
    """Una run está completa si existe su molecules.csv (lo escribe al final)."""
    return os.path.exists(os.path.join(run_dir_of(t), "molecules.csv"))


def label(t):
    if t['alg'] == "MOPSO":
        return f"MOPSO/pop{t['pop']}/run_{t['run'] + 1:02d}"
    return f"{t['alg']}[{t['cx']}+{t['mut']}]/pop{t['pop']}/run_{t['run'] + 1:02d}"


# ─── Ejecución de una run (subproceso aislado) ────────────────────────────────

def run_one(t, device, threads):
    """Lanza el script del experimento en un subproceso. Devuelve (t, ok, dt, stdout)."""
    env = dict(os.environ)
    # 1 proceso por run; limitar hilos evita sobresuscripción entre workers paralelos.
    env["OMP_NUM_THREADS"]      = str(threads)
    env["MKL_NUM_THREADS"]      = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["NUMEXPR_NUM_THREADS"]  = str(threads)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""   # sin CUDA: evita inicializar la GPU

    cmd = [PYTHON, os.path.join(ROOT, t['script']),
           "--pop_size", str(t['pop']), "--run_id", str(t['run']),
           "--device", device]
    if t['alg'] != "MOPSO":
        cmd += ["--crossover", t['cx'], "--mutation", t['mut']]

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
    """Default sensato para esta máquina.
    GPU: acotado por VRAM (~0.5 GB/proceso en 4 GB, dejando margen al escritorio).
    CPU: casi todos los núcleos (el cuello es RDKit, que paraleliza entre runs)."""
    ncpu = os.cpu_count() or 2
    return 4 if device == "cuda" else max(1, min(ncpu - 1, 11))


# ─── Resúmenes ────────────────────────────────────────────────────────────────

def generate_summaries(pop):
    print("\n" + "=" * 54 + "\n  Generando resúmenes...\n" + "=" * 54)
    for _, script in GA_ALGS:
        for cx, mut in OPERATORS:
            subprocess.run([PYTHON, script, "--pop_size", str(pop),
                            "--crossover", cx, "--mutation", mut, "--generate_summary"],
                           cwd=ROOT)
    subprocess.run([PYTHON, "experimento_mopso.py", "--pop_size", str(pop),
                    "--generate_summary"], cwd=ROOT)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Orquestador local de los experimentos MO (reemplaza train.sh/SLURM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pop",     type=int, default=300, help="Tamaño de población.")
    ap.add_argument("--n-runs",  type=int, default=20, help="Runs por configuración.")
    ap.add_argument("--device",  choices=["auto", "cpu", "cuda"], default="auto",
                    help="Dispositivo para el VAE (auto → GPU si hay CUDA).")
    ap.add_argument("-p", "--parallel", type=int, default=None,
                    help="Runs concurrentes (default: 4 en GPU, ~núcleos-1 en CPU).")
    ap.add_argument("--summary-only", action="store_true",
                    help="No corre experimentos; solo consolida y muestra los resúmenes.")
    args = ap.parse_args()

    # Line-buffering: mantiene el orden de la salida del padre con la de los
    # subprocesos (resúmenes) al redirigir a un archivo o pipe.
    sys.stdout.reconfigure(line_buffering=True)

    if args.summary_only:
        generate_summaries(args.pop)
        return

    device   = resolve_device(args.device)
    parallel = args.parallel or default_parallel(device)
    threads  = max(1, (os.cpu_count() or 1) // parallel)

    tasks   = build_tasks(args.pop, args.n_runs)
    pending = [t for t in tasks if not is_done(t)]
    total, done0 = len(tasks), len(tasks) - len(pending)

    print("=" * 54)
    print("  Experimentos MO — QED(↑) SA(↓) Lipinski(↑)")
    print(f"  Máquina        : {os.uname().nodename}  ({os.cpu_count()} núcleos)")
    print(f"  Dispositivo    : {device}")
    print(f"  Concurrencia   : {parallel} runs  ({threads} hilos/run)")
    print(f"  Total de runs  : {total}   (ya hechas: {done0}, pendientes: {len(pending)})")
    print("=" * 54)

    if not pending:
        print("Todas las runs ya estaban completas.")
        generate_summaries(args.pop)
        return

    # Pre-calcula los caches deterministas UNA sola vez (evita que los N workers los
    # reconstruyan en paralelo): SMILES de train de MOSES (parseo del CSV de 84 MB →
    # pico de RAM) y las direcciones de referencia de NSGA-III/MOEA-D (~6 s). Idempotente.
    print(f"[{time.strftime('%F %T')}] preparando caches (MOSES SMILES + ref_dirs)...", flush=True)
    subprocess.run([PYTHON, "-c",
                    "import utils_mo; "
                    "print('  MOSES:', len(utils_mo._load_moses_train_smiles()), 'SMILES'); "
                    f"print('  ref_dirs:', len(utils_mo.get_ref_dirs({args.pop})), 'dirs')"],
                   cwd=ROOT)

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

    generate_summaries(args.pop)


if __name__ == "__main__":
    main()
