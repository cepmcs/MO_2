"""
Orquestador local del análisis de SENSIBILIDAD DE HIPERPARÁMETROS.

Corre el grid completo en esta máquina (GPU para el decode del VAE + varios
experimentos en paralelo; el cuello real es RDKit en CPU). Reanudable: una run
cuenta como completa si existe su molecules.csv.

Diseño (dos preguntas separadas, presupuesto fijo de 100k evaluaciones):
  • GA (NSGA2/NSGA3/MOEAD/AGEMOEA): por cada reparto pob×gen, cada combo de
    operadores y cada (prob_cruce, prob_mutación) → 3·4·3·3 = 108 configs c/u.
  • CMOPSO: por cada reparto pob×gen y cada (elite_size, mut, vel) → 3·3·3·3 = 81
    configs.  CMOPSO no tiene w/c1/c2 (su velocidad usa coeficientes aleatorios y
    no hay pbest), así que reemplazan a esas tres perillas del grid MOPSO anterior.
Total: 4·108 + 81 = 513 configuraciones × N_RUNS semillas.

Cada perilla barrida queda codificada en el path (results/<ALG>/<slug>/run_k) y
como columna de metrics.csv; al final se consolida todo en results/all_metrics.csv.

Uso:
    python run_experiments.py                       # default: GPU, 4 en paralelo, 20 runs
    python run_experiments.py --parallel 8          # más workers (ojo con la VRAM)
    python run_experiments.py --device cpu -p 11    # máximo throughput en CPU
    python run_experiments.py --n-runs 1            # smoke test (513 runs, 1 semilla)
    python run_experiments.py --summary-only        # solo regenerar all_metrics.csv
"""

import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils_mo import ga_run_dir, cmopso_run_dir, consolidate_all, FSP3_MIN

ROOT   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable   # el python del entorno actual (nada de rutas hardcodeadas)

# ─── Espacio de hiperparámetros ───────────────────────────────────────────────
# 3 repartos de población×generación, todos = 100.000 evaluaciones.
POP_GEN   = [(100, 1000), (200, 500), (400, 250)]
# Probabilidades barridas (GA). mut = prob por-gen ≈ 1/3/8 genes en 256 dims.
CX_PROBS  = [0.7, 0.9, 1.0]
MUT_PROBS = [0.004, 0.012, 0.031]
# Combos de operadores GA: (crossover, mutation).
OPERATORS = [("sbx", "pm"), ("sbx", "gauss"), ("pcx", "pm"), ("pcx", "gauss")]

# CMOPSO: sus propias perillas.  elite_size es el tamaño al que pymoo poda su archivo
# de elites (no es exactamente el γ del paper: el archivo crece hasta pop_size y solo
# entonces se poda, así que el conjunto real oscila).  La mutación se barre POR GEN con
# los mismos valores que los GA, para que sea comparable entre las cinco familias:
# CMOPSO fija prob por-individuo y deja el por-gen en 1/n_var, y el script lo pisa.
# vel_rate no está en el paper (sus ecuaciones no acotan la velocidad); se barre como
# salvaguarda contra el «swarm explosion» en un latente de 256 dimensiones.
ELITE_SIZES = [5, 10, 25]
VEL_RATES   = [0.1, 0.2, 0.35]

GA_ALGS = [
    ("NSGA2",   "experimento_nsga2.py"),
    ("NSGA3",   "experimento_nsga3.py"),
    ("MOEAD",   "experimento_moead.py"),
    ("AGEMOEA", "experimento_agemoea.py"),
]


# ─── Definición de tareas ─────────────────────────────────────────────────────

def build_tasks(n_runs):
    """Lista de tareas del grid. Cada tarea es un dict autocontenido."""
    tasks = []
    for alg, script in GA_ALGS:
        for pop, gen in POP_GEN:
            for cx, mut in OPERATORS:
                for cxp in CX_PROBS:
                    for mutp in MUT_PROBS:
                        for run in range(n_runs):
                            tasks.append(dict(kind="ga", alg=alg, script=script,
                                              pop=pop, gen=gen, cx=cx, mut=mut,
                                              cxp=cxp, mutp=mutp, run=run))
    for pop, gen in POP_GEN:
        for es in ELITE_SIZES:
            for mutp in MUT_PROBS:
                for vel in VEL_RATES:
                    for run in range(n_runs):
                        tasks.append(dict(kind="cmopso", alg="CMOPSO",
                                          script="experimento_cmopso.py",
                                          pop=pop, gen=gen, es=es, mutp=mutp,
                                          vel=vel, run=run))
    return tasks


def run_dir_of(t):
    """Path de la run — misma fuente de verdad que usan los scripts (utils_mo)."""
    if t['kind'] == 'cmopso':
        return cmopso_run_dir(t['pop'], t['gen'], t['es'], t['mutp'], t['vel'], t['run'])
    return ga_run_dir(t['alg'], t['cx'], t['mut'], t['cxp'], t['mutp'],
                      t['pop'], t['gen'], t['run'])


def is_done(t):
    """Una run está completa si existe su molecules.csv (lo escribe al final)."""
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
    # 1 proceso por run; limitar hilos evita sobresuscripción entre workers paralelos.
    env["OMP_NUM_THREADS"]      = str(threads)
    env["MKL_NUM_THREADS"]      = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["NUMEXPR_NUM_THREADS"]  = str(threads)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""   # sin CUDA: evita inicializar la GPU

    cmd = [PYTHON, os.path.join(ROOT, t['script']),
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
    """Default sensato para esta máquina.
    GPU: acotado por VRAM (~0.5 GB/proceso en 4 GB, dejando margen al escritorio).
    CPU: casi todos los núcleos (el cuello es RDKit, que paraleliza entre runs)."""
    ncpu = os.cpu_count() or 2
    return 4 if device == "cuda" else max(1, min(ncpu - 1, 11))


def prewarm_caches():
    """Pre-calcula el cache determinista UNA sola vez (evita que los N workers lo
    reconstruyan en paralelo): los SMILES de train de MOSES.  Idempotente.

    Las direcciones de referencia ya no se cachean: con 2 objetivos son Das-Dennis
    exacto y cuestan ~1 ms, así que cada run las genera sola.  Se siguen imprimiendo
    como verificación de que salen con la dimensión y el tamaño correctos."""
    pops = sorted({p for p, _ in POP_GEN})
    code = ("import utils_mo; "
            "print('  MOSES:', len(utils_mo._load_moses_train_smiles()), 'SMILES'); "
            + "".join(f"print('  ref_dirs p{p}:', utils_mo.get_ref_dirs({p}).shape); "
                      for p in pops))
    print(f"[{time.strftime('%F %T')}] preparando caches (MOSES SMILES + ref_dirs)...", flush=True)
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

    # Line-buffering: mantiene el orden de la salida del padre con la de los subprocesos.
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
