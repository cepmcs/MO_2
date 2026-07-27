"""
Orquestador de las baselines: random, LHS y GA de suma ponderada.

Mismo presupuesto que los MOEAs (100.000 evaluaciones = 400 × 250) y las mismas
20 semillas, de modo que la comparación posterior sea pareada por semilla.

Reanudable: una corrida cuenta como completa si existe su molecules.csv.

Uso:
    python run_baselines.py
    python run_baselines.py --methods random lhs
    python run_baselines.py --n-runs 5 --parallel 2
"""

import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from experimento_baselines import baseline_run_dir, BASELINE_RESULTS_DIR

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

POP_SIZE, N_GEN = 400, 250          # 100.000 evaluaciones, igual que los MOEAs
METHODS = ['random', 'lhs', 'weighted_ga']


def tag_of(method):
    """weighted_ga guarda bajo un subdirectorio con los pesos."""
    return 'w0.333333_0.333333_0.333333' if method == 'weighted_ga' else None


def run_dir_of(t):
    return baseline_run_dir(t['method'], POP_SIZE, N_GEN, t['run'],
                            tag=tag_of(t['method']))


def is_done(t):
    return os.path.exists(os.path.join(run_dir_of(t), "molecules.csv"))


def run_one(t, device, threads):
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = str(threads)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [PYTHON, os.path.join(ROOT, "experimento_baselines.py"),
           "--method", t['method'], "--pop_size", str(POP_SIZE),
           "--n_gen", str(N_GEN), "--run_id", str(t['run']), "--device", device]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    return t, is_done(t), time.time() - t0, proc.stdout


def human(s):
    s = int(s)
    h, s = divmod(s, 3600)
    m, _ = divmod(s, 60)
    return f"{h}h {m}m" if h else f"{m}m"


def main():
    ap = argparse.ArgumentParser(description="Orquestador de baselines.")
    ap.add_argument("--methods", nargs='+', default=METHODS, choices=METHODS)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("-p", "--parallel", type=int, default=3)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    if args.summary_only:
        from utils_mo import consolidate_all
        consolidate_all(results_dir=BASELINE_RESULTS_DIR)
        return

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    threads = max(1, (os.cpu_count() or 1) // args.parallel)

    tasks = [{'method': m, 'run': r}
             for m in args.methods for r in range(args.n_runs)]
    pending = [t for t in tasks if not is_done(t)]
    done0 = len(tasks) - len(pending)

    print("=" * 58)
    print("  BASELINES — random / LHS / GA de suma ponderada")
    print(f"  Presupuesto  : {POP_SIZE} × {N_GEN} = {POP_SIZE*N_GEN:,} evaluaciones")
    print(f"  Métodos      : {', '.join(args.methods)}")
    print(f"  Dispositivo  : {device}   ({args.parallel} en paralelo)")
    print(f"  Corridas     : {len(tasks)}  (hechas: {done0}, pendientes: {len(pending)})")
    print("=" * 58)

    if not pending:
        print("Todas completas.")
        from utils_mo import consolidate_all
        consolidate_all(results_dir=BASELINE_RESULTS_DIR)
        return

    t_start, completed, failed = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(run_one, t, device, threads): t for t in pending}
        for fut in as_completed(futs):
            t, ok, dt, out = fut.result()
            completed += 1
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed else 0
            eta = human((len(pending) - completed) / rate) if rate else "…"
            print(f"[{time.strftime('%T')}] {done0+completed}/{len(tasks)} | "
                  f"{'ok' if ok else 'FALLÓ':5s} {t['method']:11s} run {t['run']+1:02d} "
                  f"({dt:.0f}s) | ETA {eta}", flush=True)
            if not ok:
                failed.append(t)
                print("  └─ " + "\n     ".join(out.strip().splitlines()[-6:]),
                      flush=True)

    print("=" * 58)
    print(f"  FIN: {len(tasks)-len(failed)}/{len(tasks)} en {human(time.time()-t_start)}")
    if failed:
        for t in failed:
            print(f"    falló: {t['method']} run {t['run']+1}")
    print("=" * 58)

    from utils_mo import consolidate_all
    consolidate_all(results_dir=BASELINE_RESULTS_DIR)


if __name__ == "__main__":
    main()
