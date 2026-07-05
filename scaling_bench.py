"""Curva de escalado: ¿cuántas runs concurrentes conviene lanzar en un nodo
antes de que la contención (memory-bound) mate la ganancia?

Método: lanza N copias del MISMO run (CPU, 1 hilo c/u) a la vez y mide cuánto se
alarga el tiempo por run vs correrlo solo.  throughput(N) = N / tiempo_por_run.
Mientras el throughput suba, más N conviene; cuando se aplana, ese es el "codo"
(tu -P en train.sh).  Pasado el codo solo añades lentitud sin ganar throughput.

Uso (correr EN EL NODO donde vas a lanzar train.sh):
    export PYTHONPATH=/ruta/al/repo          # o corre desde el dir del repo
    python scaling_bench.py sweep 300 20 /tmp/sweep 8,16,32,48,64

Cada nivel usa pocas generaciones (la contención por generación es la misma que
en la run real), así el barrido completo tarda ~pocos minutos.
"""
import os, sys, time, subprocess, statistics

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)


def worker(pop, n_gen, out_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""     # forzar CPU
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np, torch
    torch.set_num_threads(1)
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from utils_mo import (
        load_model, load_seed_mus, load_train_smiles,
        MolecularLatentProblem, LatentSampling, GenerationTracker, get_operators)
    np.random.seed(0); torch.manual_seed(0)
    model, stoi, itos, latent_dim = load_model()
    mus = load_seed_mus(model, stoi, pop, 0)
    train_smiles = load_train_smiles()
    cx, mut = get_operators('sbx', 'pm')
    problem = MolecularLatentProblem(model, stoi, itos, latent_dim)
    tracker = GenerationTracker(problem, train_smiles)
    alg = NSGA2(pop_size=pop, sampling=LatentSampling(mus),
                crossover=cx, mutation=mut, eliminate_duplicates=True)
    t0 = time.time()
    minimize(problem, alg, ('n_gen', n_gen), seed=0, verbose=False, callback=tracker)
    with open(out_file, "w") as f:
        f.write(str(time.time() - t0))
    sys.stdout.flush(); os._exit(0)


def sweep(pop, n_gen, outdir, levels, n_runs_total=340, n_gen_real=500):
    os.makedirs(outdir, exist_ok=True)
    base = None
    rows = []
    print(f"pop={pop}  n_gen(bench)={n_gen}  extrapolando a {n_gen_real} gen y "
          f"{n_runs_total} runs\n", flush=True)
    for N in levels:
        procs, outs = [], []
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""; env["OMP_NUM_THREADS"] = "1"
        for i in range(N):
            of = os.path.join(outdir, f"n{N}_i{i}.txt")
            outs.append(of)
            procs.append(subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "worker", str(pop), str(n_gen), of],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for p in procs:
            p.wait()
        per_run = statistics.mean(float(open(o).read()) for o in outs)   # compute-only
        if base is None:
            base = per_run
        per_run_real = per_run / n_gen * n_gen_real
        eff = base / per_run
        thr = N / per_run_real * 3600
        rows.append((N, per_run_real, eff, thr))
        print(f"N={N:3d}  t/run->{n_gen_real}gen={per_run_real/60:5.1f}min  "
              f"efic={eff*100:3.0f}%  throughput={thr:6.1f} runs/h", flush=True)

    print(f"\n--- Tiempo estimado para {n_runs_total} runs en 1 nodo, según N ---")
    best = max(rows, key=lambda r: r[3])
    for N, per_run_real, eff, thr in rows:
        total_h = n_runs_total / thr
        mark = "  <- máx throughput" if (N, per_run_real, eff, thr) == best else ""
        print(f"  -P {N:3d}:  {total_h*60:6.1f} min  ({total_h:.2f} h)   efic {eff*100:3.0f}%{mark}")
    print("\nRegla: elige el N más alto donde el throughput todavía sube claro y la "
          "eficiencia siga ≳70%. Ese es tu PARALLEL en train.sh.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("worker", "sweep"):
        print(__doc__); sys.exit(1)
    if sys.argv[1] == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        pop, n_gen, outdir = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        levels = [int(x) for x in sys.argv[5].split(",")]
        sweep(pop, n_gen, outdir, levels)
