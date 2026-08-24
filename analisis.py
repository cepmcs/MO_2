"""
Análisis de los experimentos: las cuatro etapas de la comparación, más la figura
de moléculas representativas.  Cada etapa lee lo que produjo la anterior y deja
sus tablas y figuras bajo plots/.

Objetivos de esta campaña: QED (↑) y SA (↓).  Fsp3 dejó de ser objetivo y entra
como restricción (Fsp3 ≥ 0.3), así que el frente es bidimensional y aparece una
métrica nueva —la factibilidad— que mide qué fracción de lo generado es
admisible.  El algoritmo de enjambre es CMOPSO, que maneja el constraint de
forma nativa, en lugar del MOPSO_CD de la campaña anterior.  Los números NO son
comparables con los de la etapa a 3 objetivos: el hipervolumen se mide sobre 2
ejes (máximo 1.21 en vez de 1.331).

  etapa1     resultados/grid/all_metrics.csv          →  plots/hiperparametros/
  etapa2     resultados/winners/                      →  plots/operadores/
  etapa3     resultados/finalistas/                   →  plots/comparacion_final/
  etapa4     resultados/finalistas/ + .../baselines/  →  plots/baselines/
  moleculas  resultados/finalistas/                   →  plots/comparacion_final/

Las 20 semillas están pareadas en todas las etapas (mismo run_id → misma
población inicial), así que los tests toman la semilla como bloque: Friedman
sobre los métodos y, si resulta significativo, las comparaciones por pares con
Wilcoxon de rangos con signo corregidas por Holm.  El resultado se resume en
grupos homogéneos: dentro de una llave el post-hoc no separa, entre llaves sí.

Uso:
    python analisis.py etapa1 [--algorithms NSGA2 CMOPSO] [--metric spacing]
    python analisis.py etapa2 [--algorithms NSGA2 MOEAD] [--metric igd_plus]
    python analisis.py etapa3 [--finalistas otra_carpeta]
    python analisis.py etapa4 [--metric igd_plus]
    python analisis.py moleculas [--out figura.png]

La carga de resultados y las gráficas comunes viven en plot_comparison.py.
"""

import io
import os
import re
import math
import glob
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# Importar plot_comparison también fija el estilo global de matplotlib.
import plot_comparison as pc

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Entradas de cada etapa, todas bajo resultados/.  Los experimentos escriben en
# el cluster a results/ y results_baselines/ (ver utils_mo.RESULTS_DIR y
# experimento_baselines.BASELINE_RESULTS_DIR); lo que baja al PC vive acá:
#
#   grid/        copia ligera de results/ del cluster (sin convergence.csv ni
#                all_molecules.csv.gz), con el all_metrics.csv del grid completo
#   winners/     las 17 configuraciones que ganaron su bloque en la etapa 1
#   finalistas/  symlinks relativos a la ganadora de cada algoritmo en winners/
#   baselines/   copia de results_baselines/ del cluster
RESULTADOS_DIR = os.path.join(ROOT_DIR, "resultados")
METRICS_CSV    = os.path.join(RESULTADOS_DIR, "grid", "all_metrics.csv")
WINNERS_DIR    = os.path.join(RESULTADOS_DIR, "winners")
FINALISTAS_DIR = os.path.join(RESULTADOS_DIR, "finalistas")
BASELINES_DIR  = os.path.join(RESULTADOS_DIR, "baselines")

# Salidas, todas bajo plots/.  El componente 'operadores' de la etapa 2 es
# significativo: plot_comparison deduce de él el nombre del algoritmo para
# titular las figuras de frentes.
PLOTS_DIR      = os.path.join(ROOT_DIR, "plots")
OUT_HP         = os.path.join(PLOTS_DIR, "hiperparametros")
OUT_OPERADORES = os.path.join(PLOTS_DIR, "operadores")
OUT_ALGORITMOS = os.path.join(PLOTS_DIR, "comparacion_final")
OUT_BASELINES  = os.path.join(PLOTS_DIR, "baselines")
# El frente conjunto va aparte: no compara algoritmos entre sí como el resto de
# la etapa 3, sino que caracteriza qué moléculas sobreviven al unirlos y de dónde
# salen.  Son preguntas distintas y conviene que no se mezclen en la lectura.
OUT_FRENTE     = os.path.join(PLOTS_DIR, "frente_conjunto")

# Nombres para el documento (los directorios usan la forma corta).
DISPLAY = {'NSGA2': 'NSGA-II', 'NSGA3': 'NSGA-III', 'MOEAD': 'MOEA/D',
           'AGEMOEA': 'AGE-MOEA', 'CMOPSO': 'CMOPSO',
           'RANDOM': 'Aleatorio', 'WEIGHTED_GA': 'GA ponderado',
           'SCREENING': 'Cribado MOSES', 'HILL_CLIMBER': 'Escalador'}

# El algoritmo de enjambre de esta etapa.  Se nombra una sola vez: es el que no
# tiene operadores de cruce/mutación y por eso queda fuera de la comparación de
# operadores, del pool por familias y de los factores GA.  CMOPSO reemplazó al
# MOPSO_CD anterior porque maneja el constraint de forma nativa.
PSO_ALG = 'CMOPSO'


# ═══════════════════════════════════════════════════════════════════════════
#   Utilidades compartidas
# ═══════════════════════════════════════════════════════════════════════════

_latex_escape = pc._latex_escape


def _num(x, dec):
    """Número con el separador decimal del documento (pc.SEP_DECIMAL)."""
    return f'{x:.{dec}f}'.replace('.', pc.SEP_DECIMAL)


def _fmt_p(p):
    if p is None or pd.isna(p):
        return '---'
    return f'$<$0{pc.SEP_DECIMAL}001' if p < 1e-3 else _num(p, 3)


def fmt_groups(groups):
    """'{A, B} $>$ {C}' con los nombres de presentación.

    Escapa los nombres: los combos de operadores llevan guion bajo (pcx\\_pm) y
    sin escapar rompen la compilación."""
    return ' $>$ '.join(
        '\\{' + ', '.join(_latex_escape(DISPLAY.get(x, x)) for x in g) + '\\}'
        for g in groups)


def holm(pvals):
    """Corrección de Holm-Bonferroni.  Devuelve los p ajustados en el orden de
    entrada."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvals[idx]
        prev = max(prev, min(val, 1.0))    # monotonía
        adj[idx] = prev
    return adj


def rank_biserial(x, y):
    """Correlación rango-biserial de pares emparejados (Kerby, 2014).

    Es el tamaño de efecto que acompaña al Wilcoxon de rangos con signo: sobre
    los rangos de |x - y|, la suma de los que favorecen a x menos la de los que
    favorecen a y, dividida por el total.  Va de -1 a +1; el signo dice quién
    gana y el valor absoluto qué fracción de la evidencia lo respalda.

    Satura en ±1 cuando todos los pares van en la misma dirección, así que dice
    que el efecto es unánime, no cuán grande es en unidades del indicador.
    Verificado contra pingouin.wilcoxon()['RBC'].
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]                      # los empates exactos no aportan rango
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


# No se etiqueta la magnitud (pequeño/mediano/grande): los umbrales que circulan
# se derivan de convertir los cortes de Cohen a distintas escalas y no coinciden
# entre fuentes, ninguna pensada para la versión de pares emparejados.  El valor
# se reporta crudo.


def compare_indicator(get_values, labels, col):
    """Friedman sobre los métodos con la semilla como bloque + post-hoc.

    Devuelve dict con el p de Friedman, la mediana de cada método y la lista de
    comparaciones por pares con su p corregido por Holm; None si falta algún
    método o hay menos de 3 semillas."""
    cols = {}
    for lab in labels:
        v = get_values(lab, col)
        if v is None:
            return None
        cols[lab] = np.asarray(v, dtype=float)

    n = min(len(v) for v in cols.values())
    if n < 3:
        return None
    M = np.column_stack([cols[lab][:n] for lab in labels])   # semillas × métodos
    k = M.shape[1]

    try:
        p_omni = float(stats.friedmanchisquare(*[M[:, j] for j in range(k)]).pvalue)
    except ValueError:
        p_omni = np.nan

    pairs, raw = [], []
    for i, j in itertools.combinations(range(k), 2):
        try:
            p = float(stats.wilcoxon(M[:, i], M[:, j]).pvalue)
        except ValueError:      # todas las diferencias son cero
            p = 1.0
        pairs.append((labels[i], labels[j]))
        raw.append(p)
    adj = holm(raw) if raw else []

    return {'p_omnibus': p_omni,
            'medians': {lab: float(np.median(cols[lab][:n])) for lab in labels},
            'pairs': [{'a': a, 'b': b, 'p_raw': pr, 'p_holm': pa}
                      for (a, b), pr, pa in zip(pairs, raw, adj)]}


def homogeneous_groups(res, labels, medians, higher_better):
    """Grupos de métodos que el post-hoc no logra separar, ordenados de mejor a
    peor.  Notación estándar para resumir las comparaciones en una línea."""
    order = sorted(labels, key=lambda l: -medians[l] if higher_better else medians[l])
    sep = {(p['a'], p['b']): p['p_holm'] < 0.05 for p in res['pairs']}

    def differ(a, b):
        return sep.get((a, b), sep.get((b, a), False))

    groups, cur = [], [order[0]]
    for lab in order[1:]:
        if any(differ(lab, m) for m in cur):
            groups.append(cur)
            cur = [lab]
        else:
            cur.append(lab)
    groups.append(cur)
    return groups


def _write_tex(lines, path, msg=None):
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"  ✓ {msg or os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 1 — selección de hiperparámetros
#
#   De las 513 configuraciones del grid elige 17: la mejor de cada combinación
#   de operadores en los 4 GA, más la mejor global de CMOPSO.  Gana la de menor
#   rango medio de hipervolumen, rankeando dentro de cada una de las 20 semillas
#   (que están pareadas: la población inicial se muestrea con random_state=run_id).
#
#   Los operadores no se testean acá: se barren dentro de cada bloque y su
#   comparación es la etapa 2, ya con la configuración de cada combo afinada.
#
#   Deja tres archivos, todos a nivel de plots/hiperparametros/: la figura de
#   selección con los cinco algoritmos, y selected_configs en .csv (lo lee
#   run_experiments.py) y .tex.
# ═══════════════════════════════════════════════════════════════════════════

# CMOPSO no tiene operadores; se colorea por tamaño del archivo de elites, que es
# la perilla propia de su mecanismo de selección de líderes (reemplaza a la
# inercia w del MOPSO anterior, que en CMOPSO no existe).
ELITE_COLORS = {5: '#9ECAE1', 10: '#4292C6', 25: '#08519C'}

# columna: (etiqueta, higher_better)
HP_METRICS = {
    'hypervolume': ('Hipervolumen', True),
    'spacing':     ('Espaciamiento', False),
    'n_pareto':    ('Tamaño de Pareto', True),
    'validity':    ('Validez', True),
    # Fracción de las moléculas válidas que cumplen el constraint de saturación.
    # Es la métrica nueva de esta etapa: mide cuánto del presupuesto se gasta
    # fuera de la región admisible, algo que no existía con Fsp3 como objetivo.
    'feasibility': ('Factibilidad', True),
    'novelty':     ('Novedad', True),
    'best_sa':     ('Mejor SA', False),
    'time_sec':    ('Tiempo (s)', False),
}

# 'budget' agrupa pop_size×n_gen: están acoplados por el presupuesto fijo de 100k
# evaluaciones.
FACTORS_GA = ['budget', 'crossover', 'mutation', 'cx_prob', 'mut_prob']
# Las perillas de CMOPSO no son las del MOPSO anterior: su ecuación de velocidad
# usa coeficientes aleatorios por dimensión y no hay pbest, así que w/c1/c2
# desaparecen y en su lugar se barren el archivo de elites, la mutación por-gen y
# el tope de velocidad (ver run_experiments.py).
FACTORS_PSO = ['budget', 'elite_size', 'mut_prob', 'vel_rate']

COMBO_FACTORS = ['crossover', 'mutation']
SUB_FACTORS_GA = ['budget', 'cx_prob', 'mut_prob']

OPERATOR_COLORS = {
    'pcx/pm':    '#1F4E79',
    'pcx/gauss': '#6FA8DC',
    'sbx/pm':    '#B45F06',
    'sbx/gauss': '#F6B26B',
}
# Combos tal como aparecen en el CSV del grid (la etapa 2 los lee de winners/,
# donde son nombres de directorio con guion bajo: ver COMBO_DIRS).
HP_COMBOS = ['pcx/pm', 'pcx/gauss', 'sbx/pm', 'sbx/gauss']

def factors_for(alg):
    return FACTORS_PSO if alg == PSO_ALG else FACTORS_GA


# ─── Carga y selección ───────────────────────────────────────────────────────

def load_grid(csv_path):
    """Carga el CSV consolidado y añade la columna derivada 'budget'."""
    df = pd.read_csv(csv_path)
    df['budget'] = (df['pop_size'].astype(int).astype(str) + '×'
                    + df['n_gen'].astype(int).astype(str))
    return df


def run_matrix(g, factors, metric):
    """Matriz (configuraciones × semillas) de la métrica, sin las configs
    incompletas."""
    M = g.pivot_table(index=factors, columns='run', values=metric)
    complete = M.notna().all(axis=1)
    if not complete.all():
        print(f"  ⚠ {int((~complete).sum())} config(s) con semillas faltantes; se omiten")
    return M[complete]


def select_config(M, higher_better):
    """Rango medio de cada configuración, rankeando dentro de cada semilla
    (rango 1 = mejor de esa semilla).  Devuelve (elegida, rangos)."""
    A = -M.values if higher_better else M.values
    R = np.apply_along_axis(stats.rankdata, 0, A)
    ranks = pd.Series(R.mean(axis=1), index=M.index)
    return ranks.idxmin(), ranks


# Nombre corto de cada perilla en las etiquetas de configuración.  Las de CMOPSO
# aparecen acá porque su nombre de columna es largo y la etiqueta va a la tabla
# del documento: 'elite_size=10' ocuparía una columna entera.
FACTOR_ABBR = {'cx_prob': 'cx', 'mut_prob': 'mut',
               'elite_size': 'elite', 'vel_rate': 'vel'}


def config_label(cfg, factors):
    """Etiqueta compacta, p. ej. '400×250 pcx/pm cx=1 mut=0.031' o, en CMOPSO,
    '400×250 elite=10 mut=0.031 vel=0.2'."""
    cfg = cfg if isinstance(cfg, tuple) else (cfg,)
    parts, cx, mu = [], None, None
    for f, v in zip(factors, cfg):
        if f == 'budget':
            parts.append(str(v))
        elif f == 'crossover':
            cx = v
        elif f == 'mutation':
            mu = v
        else:
            parts.append(f'{FACTOR_ABBR.get(f, f)}={v:g}')
    if cx is not None:
        parts.insert(1, f'{cx}/{mu}' if mu is not None else cx)
    return ' '.join(parts)


# ─── Gráfica ─────────────────────────────────────────────────────────────────

def _panel_seleccion(ax, g, alg, metric, chosen, sub_factors, por_combo):
    """Un algoritmo: cada configuración del grid como un punto (validez contra
    la métrica de selección, ambas medianas sobre las 20 semillas) y la elegida
    de cada bloque resaltada.  El color separa las combinaciones de operadores;
    en CMOPSO, que no tiene operadores, el tamaño del archivo de elites."""
    fs = [f for f in factors_for(alg) if g[f].notna().any()]
    m = g.groupby(fs, observed=True)[[metric, 'validity']].median()

    if por_combo:
        key = (m.index.get_level_values('crossover').astype(str) + '/'
               + m.index.get_level_values('mutation').astype(str))
        groups = [(c, OPERATOR_COLORS[c]) for c in HP_COMBOS if c in set(key)]
    else:
        # elite_size llega del CSV como float (la columna trae NaN en las filas
        # de los GA y pandas la promueve), así que la clave del mapa de colores
        # se busca por su valor entero.
        key = m.index.get_level_values('elite_size')
        presentes = {int(v) for v in set(key)}
        groups = [(e, c) for e, c in sorted(ELITE_COLORS.items())
                  if e in presentes]
        key = pd.Index([int(v) for v in key])

    for name, color in groups:
        sel = key == name
        ax.scatter(m.loc[sel, 'validity'], m.loc[sel, metric], s=30, alpha=0.65,
                   color=color, linewidths=0, zorder=2)

    for name, b in chosen.items():
        cfg = b['cfg'] if isinstance(b['cfg'], tuple) else (b['cfg'],)
        lv = dict(zip(sub_factors, cfg))
        if name:
            lv['crossover'], lv['mutation'] = name.split('/')
        r = m.loc[tuple(lv[f] for f in fs)]
        color = (OPERATOR_COLORS[name] if por_combo
                 else ELITE_COLORS.get(int(lv.get('elite_size', -1)), '#333333'))
        ax.scatter(r['validity'], r[metric], s=170, color=color,
                   edgecolors='black', linewidths=1.6, zorder=4)

    ax.margins(x=0.07, y=0.10)
    ax.set_title(DISPLAY.get(alg, alg), fontsize=13)


def _celda_leyenda(ax, con_operadores, con_pso):
    """La celda libre de la grilla, usada como leyenda de toda la figura."""
    ax.axis('off')
    punto = lambda color, **kw: plt.Line2D([], [], marker='o', linestyle='none',
                                           markersize=9, markerfacecolor=color,
                                           markeredgewidth=0, **kw)
    bloques = []
    if con_operadores:
        bloques.append(('Operadores (algoritmos genéticos)',
                        [punto(OPERATOR_COLORS[c], label=c) for c in HP_COMBOS]))
    if con_pso:
        bloques.append((f'{PSO_ALG}: archivo de elites',
                        [punto(c, label=f'elite = {e:g}')
                         for e, c in sorted(ELITE_COLORS.items())]))
    bloques.append((None, [plt.Line2D([], [], marker='o', linestyle='none',
                                      markersize=13, markerfacecolor='#bbbbbb',
                                      markeredgecolor='black',
                                      markeredgewidth=1.6,
                                      label='Configuración seleccionada')]))

    # Los bloques se apilan de arriba abajo repartiendo la celda por filas
    # (una por entrada, más una por título), para que entren siempre.
    filas = [len(h) + (1 if t else 0) for t, h in bloques]
    alto = min(0.085, 0.92 / max(sum(filas), 1))
    y = 0.97
    for (titulo, handles), n in zip(bloques, filas):
        leg = ax.legend(handles=handles, title=titulo, loc='upper center',
                        bbox_to_anchor=(0.5, y), frameon=False, fontsize=11,
                        handletextpad=0.6, borderaxespad=0)
        if titulo:
            leg.get_title().set_fontweight('bold')
        ax.add_artist(leg)
        y -= alto * (n + 0.7)


def plot_seleccion_grid(df, algs, metric, out_dir, per_alg):
    """Los cinco algoritmos en una sola imagen: un panel por algoritmo con todo
    su grid de configuraciones y la elegida de cada bloque resaltada.

    Es la única figura de la etapa: muestra a la vez el compromiso entre validez
    y calidad del frente, dónde cae cada familia de operadores y qué punto se
    llevó cada bloque.  El eje de validez es común a los cinco paneles para que
    la posición horizontal sea comparable; el de la métrica es propio de cada
    uno, porque los rangos difieren y compartirlo aplastaría los paneles."""
    label, _ = HP_METRICS[metric]
    if metric == 'validity':
        return
    algs = [a for a in algs if a in per_alg]
    if not algs:
        return

    ncols = 3
    nrows = math.ceil((len(algs) + 1) / ncols)     # +1: la celda de la leyenda
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.7 * ncols, 4.9 * nrows),
                             squeeze=False, sharex=True)
    axes = axes.flatten()

    for ax, alg in zip(axes, algs):
        d = per_alg[alg]
        _panel_seleccion(ax, df[df['algorithm'] == alg], alg, metric,
                         d['blocks'], d['sub_factors'], d['por_combo'])
        # sharex esconde las marcas de los paneles con otro axes debajo; acá
        # ese "otro axes" puede ser la celda de la leyenda, que no lleva eje.
        ax.tick_params(labelbottom=True)

    _celda_leyenda(axes[len(algs)],
                   con_operadores=any(per_alg[a]['por_combo'] for a in algs),
                   con_pso=any(not per_alg[a]['por_combo'] for a in algs))
    for ax in axes[len(algs) + 1:]:
        ax.set_visible(False)

    fig.supxlabel('Validez (fracción de moléculas válidas) →', fontsize=12,
                  y=0.042)
    fig.supylabel(f'{label} →', fontsize=12)
    fig.suptitle(f'Selección de hiperparámetros: {label.lower()} contra validez',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0.01, 0.02, 1, 0.98])
    fname = f'{metric}_vs_validity.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


# Combinaciones de operadores tal como aparecen en el grid, y su codificación:
# el color lleva el cruce y la trama la mutación, de modo que se distinguen las
# cuatro sin gastar cuatro tonos y sin depender del color en impresión.
EFECTO_COMBOS = [('pcx', 'pm'), ('pcx', 'gauss'), ('sbx', 'pm'), ('sbx', 'gauss')]
EFECTO_TRAMA = {'pm': None, 'gauss': '///'}
# Naranja y azul de Okabe-Ito (los mismos con que plot_comparison separa PCX de
# SBX) y el rojo de CMOPSO oscurecido: el rojo puro queda cerca del naranja en
# pantallas de gama amplia, y bajar la luminosidad los separa igual.
EFECTO_COLOR = {'pcx': '#D55E00', 'sbx': '#0072B2', 'pso': '#B01818'}

# Perillas de cada familia de algoritmo: (columna, etiqueta corta).
EFECTO_F_GA = [('budget', 'pob$\\times$gen'), ('cx_prob', '$P$(cruce)'),
               ('mut_prob', '$P$(mut.)')]
# CMOPSO comparte con los GA la mutación por-gen —se barre con los mismos tres
# valores justamente para que el efecto sea comparable entre familias— y suma dos
# perillas propias: el archivo de elites y el tope de velocidad.
EFECTO_F_PSO = [('budget', 'pob$\\times$gen'), ('elite_size', 'elites'),
                ('mut_prob', '$P$(mut.)'), ('vel_rate', '$v_{\\max}$')]


def _efecto_factor(g, factor, metric):
    """Diferencia entre el mejor y el peor nivel del factor, en mediana de la
    métrica.  Se calcula siempre DENTRO de una combinación de operadores: sobre
    el grid entero el número se infla, porque el cruce domina y cada nivel queda
    con una distribución bimodal cuya mediana se mueve por el reparto entre modos
    y no por el factor."""
    m = g.groupby(factor, observed=True)[metric].median()
    return float(m.max() - m.min()) if len(m) > 1 else np.nan


def plot_efectos_hp(df, metric, out_dir):
    """Cuánto mueve cada hiperparámetro a la métrica, un panel por algoritmo.

    Responde la pregunta que deja abierta el grid —barrimos 513 configuraciones,
    ¿cuál perilla importó?— y complementa a la figura de selección, que muestra
    dónde cae cada configuración pero no qué factor explica la dispersión.  El
    eje vertical es común para que las alturas sean comparables entre paneles."""
    label, _ = HP_METRICS[metric]
    algs = [a for a in GA_ALGS if a in set(df['algorithm'])]
    con_pso = PSO_ALG in set(df['algorithm'])
    if not algs:
        return

    n = len(algs) + (1 if con_pso else 0)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.6), squeeze=False,
                             sharey=True)
    axes = axes[0]
    ancho = 0.20

    for ax, alg in zip(axes, algs):
        g = df[df['algorithm'] == alg]
        x = np.arange(len(EFECTO_F_GA))
        for k, (cx, mu) in enumerate(EFECTO_COMBOS):
            sel = g[(g['crossover'] == cx) & (g['mutation'] == mu)]
            ax.bar(x + (k - 1.5) * ancho,
                   [_efecto_factor(sel, f, metric) for f, _ in EFECTO_F_GA],
                   ancho, color=EFECTO_COLOR[cx], hatch=EFECTO_TRAMA[mu],
                   edgecolor='white', linewidth=0.6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([e for _, e in EFECTO_F_GA], fontsize=9)
        ax.set_title(DISPLAY.get(alg, alg), fontsize=11, fontweight='bold', pad=8)

    if con_pso:
        ax = axes[len(algs)]
        g = df[df['algorithm'] == PSO_ALG]
        x = np.arange(len(EFECTO_F_PSO))
        ax.bar(x, [_efecto_factor(g, f, metric) for f, _ in EFECTO_F_PSO], 0.62,
               color=EFECTO_COLOR['pso'], edgecolor='white', linewidth=0.6,
               zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([e for _, e in EFECTO_F_PSO], fontsize=9)
        ax.set_title(DISPLAY.get(PSO_ALG, PSO_ALG), fontsize=11,
                     fontweight='bold', pad=8)

    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        for lado in ('top', 'right'):
            ax.spines[lado].set_visible(False)

    axes[0].set_ylabel(f'Efecto sobre {label.lower()}', fontsize=10)
    manos = [plt.Rectangle((0, 0), 1, 1, facecolor=EFECTO_COLOR[cx],
                           hatch=EFECTO_TRAMA[mu], edgecolor='white',
                           linewidth=0.6) for cx, mu in EFECTO_COMBOS]
    etiqs = [f'{cx.upper()} + {"PM" if mu == "pm" else "gaussiana"}'
             for cx, mu in EFECTO_COMBOS]
    if con_pso:
        manos.append(plt.Rectangle((0, 0), 1, 1, facecolor=EFECTO_COLOR['pso']))
        etiqs.append(f'{PSO_ALG} (sin operadores)')
    fig.legend(manos, etiqs, loc='lower center', ncol=len(etiqs), frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.13))

    fig.suptitle(f'Efecto de los hiperparámetros sobre {label.lower()}',
                 fontsize=13, fontweight='bold', y=1.04)
    fig.tight_layout()
    fname = f'efectos_{metric}.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


# ─── Tablas ──────────────────────────────────────────────────────────────────

def write_selection_summary(per_alg, metric, out_dir):
    """CSV y tabla LaTeX de las configuraciones seleccionadas.  El CSV usa los
    nombres de columna que consume run_experiments.py."""
    label, _ = HP_METRICS[metric]
    rows = []
    for alg in [a for a in pc.ALGORITHM_ORDER if a in per_alg]:
        d = per_alg[alg]
        for name, b in d['blocks'].items():
            lv = dict(zip(d['sub_factors'], b['cfg']))
            if d['por_combo'] and name:
                lv['crossover'], lv['mutation'] = name.split('/')
            pop, gen = str(lv['budget']).split('×')
            row = {'algorithm': alg, 'operators': name or '',
                   'pop_size': int(pop), 'n_gen': int(gen),
                   'config': b['label'], 'avg_rank': b['rank'],
                   'mean': b['mean'], 'std': b['std'],
                   'n_configs': b['n_configs']}
            for f in ('crossover', 'mutation', 'cx_prob', 'mut_prob',
                      'elite_size', 'vel_rate'):
                if f in lv:
                    row[f] = lv[f]
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, 'selected_configs.csv'), index=False)
    print(f"  ✓ selected_configs.csv  ({len(out)} configuraciones)")

    lines = [
        r'\begin{table}[htbp]', r'\centering', r'\small',
        f'\\caption{{Configuraciones seleccionadas: la mejor de cada combinación '
        f'de operadores en los algoritmos genéticos, y la mejor global en '
        f'{_latex_escape(DISPLAY.get(PSO_ALG, PSO_ALG))}. '
        f'Gana la de menor rango medio de {_latex_escape(label)}, rankeando las '
        f'configuraciones dentro de cada una de las 20 semillas.}}',
        r'\label{tab:hp_seleccionadas}',
        r'\begin{tabular}{lllcc}', r'\toprule',
        r'Algoritmo & Operadores & Configuración & Rango medio & '
        r'$\mu \pm \sigma$ \\',
        r'\midrule',
    ]
    prev = None
    for _, r in out.iterrows():
        alg_cell = '' if r['algorithm'] == prev else _latex_escape(r['algorithm'])
        if alg_cell and prev is not None:
            lines.append(r'\midrule')
        prev = r['algorithm']
        # La etiqueta de configuración trae los hiperparámetros con punto (viene
        # de config_label, que también alimenta el CSV); acá se ajusta al
        # separador del documento.
        cfg = re.sub(r'(\d)\.(\d)', rf'\1{pc.SEP_DECIMAL}\2',
                     _latex_escape(r['config']))
        lines.append(
            f"{alg_cell} & {_latex_escape(r['operators']) or '---'} & "
            f"{cfg} & {_num(r['avg_rank'], 2)} & "
            f"{_num(r['mean'], 4)} $\\pm$ {_num(r['std'], 4)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, 'selected_configs.tex'))


# ─── Orquestación ────────────────────────────────────────────────────────────

def analyze_algorithm(g, alg, metric):
    """Elige la mejor configuración de cada combinación de operadores (la mejor
    global en CMOPSO, que no tiene operadores)."""
    factors = [f for f in factors_for(alg) if g[f].notna().any()]
    _, higher = HP_METRICS[metric]
    por_combo = set(COMBO_FACTORS).issubset(factors)
    sub_factors = SUB_FACTORS_GA if por_combo else factors

    print(f"\n{'─'*66}")
    print(f"  {alg}")
    print(f"{'─'*66}")

    if por_combo:
        blocks = [(f'{cx}/{mu}', gg)
                  for (cx, mu), gg in g.groupby(COMBO_FACTORS, observed=True)]
        blocks.sort(key=lambda t: HP_COMBOS.index(t[0])
                    if t[0] in HP_COMBOS else len(HP_COMBOS))
    else:
        blocks = [(None, g)]

    chosen = {}
    for name, gg in blocks:
        M = run_matrix(gg, sub_factors, metric)
        if M.shape[0] < 2:
            continue
        best, ranks = select_config(M, higher)
        vals = M.loc[best].values
        chosen[name] = {'cfg': best, 'label': config_label(best, sub_factors),
                        'rank': float(ranks.loc[best]),
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals, ddof=1)),
                        'n_configs': M.shape[0]}
        print(f"  {(name or 'todas'):11s} {chosen[name]['label']}")
        print(f"              rango {chosen[name]['rank']:.2f} de {M.shape[0]} "
              f"configs   μ {chosen[name]['mean']:.5f} ± "
              f"{chosen[name]['std']:.5f}")

    if not chosen:
        print("  ⚠ sin configuraciones completas; se omite")
        return None

    return {'blocks': chosen, 'factors': factors, 'sub_factors': sub_factors,
            'por_combo': por_combo}


def etapa1(args):
    if not os.path.exists(args.csv):
        print(f"No existe {args.csv}.\n"
              f"  En el cluster:  python run_experiments.py --summary-only\n"
              f"  Sobre una copia ya bajada:  python -c "
              f"\"from utils_mo import consolidate_all; "
              f"consolidate_all('{os.path.dirname(args.csv)}')\"")
        return

    df = load_grid(args.csv)
    algs = args.algorithms or [a for a in pc.ALGORITHM_ORDER
                               if a in set(df['algorithm'])]

    print(f"\n{'='*66}")
    print("  ETAPA 1a — SELECCIÓN DE HIPERPARÁMETROS")
    print(f"  Datos: {args.csv}  ({len(df)} ejecuciones)")
    print(f"  Métrica: {HP_METRICS[args.metric][0]} "
          f"({'↑' if HP_METRICS[args.metric][1] else '↓'})")
    print("  Criterio: menor rango medio, rankeando dentro de cada semilla")
    print(f"  Algoritmos: {', '.join(algs)}")
    print(f"{'='*66}")

    os.makedirs(args.out, exist_ok=True)
    per_alg = {}
    for alg in algs:
        g = df[df['algorithm'] == alg].copy()
        if g.empty:
            print(f"\n  ⚠ {alg}: sin datos")
            continue
        r = analyze_algorithm(g, alg, args.metric)
        if r:
            per_alg[alg] = r

    if per_alg:
        print(f"\n{'─'*66}\n  Salidas\n{'─'*66}")
        plot_seleccion_grid(df, algs, args.metric, args.out, per_alg)
        plot_efectos_hp(df, args.metric, args.out)
        write_selection_summary(per_alg, args.metric, args.out)

    print(f"\n{'='*66}")
    print(f"  ✅ Listo: {args.out}")
    print(f"{'='*66}\n")


# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 2 — comparación de combinaciones de operadores, por algoritmo
#
#   Lee winners/<ALG>/<cruce>_<mutacion>/<config>/run_XX/ (lo que ganó su bloque
#   en la etapa 1) y compara los 4 combos entre sí, un reporte por algoritmo.
# ═══════════════════════════════════════════════════════════════════════════

GA_ALGS = ['NSGA2', 'NSGA3', 'MOEAD', 'AGEMOEA']

# Los combos como nombres de directorio bajo winners/ (con guion bajo).
COMBO_DIRS = ['pcx_pm', 'pcx_gauss', 'sbx_pm', 'sbx_gauss']

# (columna, etiqueta, mayor_es_mejor).  Decide el hipervolumen; el resto está
# para poder rehacer la etapa con otro criterio desde --metric.
OP_INDICATORS = [
    ('hypervolume', 'Hipervolumen',      True),
    ('igd_plus',    'IGD$^+$',           False),
    ('epsilon',     r'$\epsilon^+$',     False),
    ('spacing',     'Espaciamiento',     False),
    ('n_pareto',    'Tamaño de Pareto',  True),
    ('validity',    'Validez',           True),
    ('feasibility', 'Factibilidad',      True),
    ('uniqueness',  'Unicidad',          True),
]


# ─── Comparación de operadores ───────────────────────────────────────────────
#
#   El grid es un 2×2 (cruce × mutación) y los cuatro combos se contrastan de una
#   vez: Friedman con las 20 semillas como bloques y las seis comparaciones por
#   pares con Wilcoxon corregidas por Holm, resumidas en grupos homogéneos.
#   Decide UN indicador, el que llega en --metric (por defecto el hipervolumen);
#   dentro del grupo ganador, que por definición el post-hoc no separa, se
#   conserva la mutación estándar.
#
#   La contribución al frente conjunto se evaluó como criterio de decisión
#   alternativo y se descartó (los ganadores quedaron por hipervolumen: pcx_pm en
#   los cuatro AG).  De ella esta etapa emite solo la figura del frente conjunto,
#   que muestra el mecanismo; el contraste con test propio quedó en la etapa 3,
#   sobre el pool de candidatos.

MUT_STD = 'pm'          # la estándar; se conserva cuando el test no separa


def write_tabla_operadores(res, groups, labels, alg, out_dir, get_values, col):
    """La tabla de la etapa: las seis comparaciones por pares entre los cuatro
    combos, cada una con su $p$ corregido y su tamaño de efecto.

    Se listan las seis y no solo las que involucran al seleccionado: la de las
    dos mutaciones dentro de SBX no interviene en la decisión pero sí es
    significativa en algunos algoritmos, y omitirla dejaría ese resultado
    únicamente insinuado por los grupos homogéneos.

    El $p$ dice si hay diferencia y el $r_{rb}$ de qué tamaño; sin el segundo,
    dos pares igualmente 'no significativos' parecen equivalentes cuando no lo
    son.  Quién gana se resuelve en los grupos del caption, no par a par.

    Devuelve el campeón: dentro del grupo ganador se conserva la mutación
    estándar si está, porque el post-hoc no separa a sus miembros."""
    orden = sorted(labels, key=lambda l: -res['medians'][l])
    top = groups[0]
    campeon = next((l for l in top if l.endswith(f'_{MUT_STD}')),
                   max(top, key=lambda l: res['medians'][l]))
    vals = {l: np.asarray(get_values(l, col), dtype=float) for l in labels}
    medianas = ', '.join(f'{_latex_escape(l)} {_num(res["medians"][l], 4)}'
                         for l in orden)

    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparación de operadores en '
        f'{_latex_escape(DISPLAY.get(alg, alg))} sobre hipervolumen.  Test de '
        f'Friedman con las 20 semillas como bloques '
        f'($p$ = {_fmt_p(res["p_omnibus"])}), seguido de las comparaciones por '
        f'pares con Wilcoxon de rangos con signo y corrección de Holm '
        f'($\\alpha = {_num(0.05, 2)}$); $r_{{rb}}$ es la correlación '
        f'rango-biserial de '
        f'pares emparejados, con signo positivo cuando gana el primero del par.  '
        f'Medianas: {medianas}.  Grupos homogéneos, de mejor a peor: '
        f'{fmt_groups(groups)}.  Como el post-hoc no separa a los miembros del '
        f'grupo ganador, se selecciona el de mutación polinomial por ser la '
        f'estándar ({_latex_escape(campeon)}).}}',
        f'\\label{{tab:ops_{alg.lower()}}}',
        r'\begin{tabular}{lrrl}', r'\toprule',
        r'Par & $p$ (Holm) & $r_{rb}$ & Mejor \\', r'\midrule',
    ]
    for p in res['pairs']:
        a, b = p['a'], p['b']
        r = rank_biserial(vals[a], vals[b])
        # Donde el post-hoc no separa no se declara ganador: la mediana ordena
        # igual, pero llamarlo «mejor» afirmaría una diferencia que no hay.
        mejor = (_latex_escape(a if res['medians'][a] > res['medians'][b] else b)
                 if p['p_holm'] < 0.05 else '---')
        lines.append(
            f"{_latex_escape(a)} vs {_latex_escape(b)} & "
            f"{_fmt_p(p['p_holm'])} & "
            f"{('$+$' if r >= 0 else '$-$') + _num(abs(r), 3)} & "
            f"{mejor} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, f'operadores_{alg}.tex'))

    print(f"  grupos: " + ' > '.join('{' + ', '.join(g) + '}' for g in groups))
    print(f"  campeón: {campeon}")
    for p in res['pairs']:
        r = rank_biserial(vals[p['a']], vals[p['b']])
        print(f"    {p['a']:10s} vs {p['b']:11s} p={p['p_holm']:8.4f}  "
              f"r_rb={r:+.3f}")
    return campeon


def _test_aporte(por_grupo, runs):
    """Contraste sobre el % aportado por semilla.  Con dos grupos alcanza el
    Wilcoxon pareado; con más hace falta el mismo Friedman + Holm que el resto
    de la comparación, resumido en grupos homogéneos.

    Devuelve (texto para el caption, dict de resumen) o (None, {})."""
    grupos = list(por_grupo)
    if len(grupos) < 2 or len(runs) < 3:
        return None, {}

    if len(grupos) == 2:
        a, b = grupos
        try:
            p = float(stats.wilcoxon(por_grupo[a], por_grupo[b]).pvalue)
        except ValueError:
            p = 1.0
        g = a if por_grupo[a].mean() > por_grupo[b].mean() else b
        otro = b if g == a else a
        txt = (f'  Repitiendo el cálculo dentro de cada semilla, '
               f'{_latex_escape(g)} aporta {_num(por_grupo[g].mean(), 1)}\\% $\\pm$ '
               f'{_num(por_grupo[g].std(ddof=1), 1)} frente a '
               f'{_num(por_grupo[otro].mean(), 1)}\\% $\\pm$ '
               f'{_num(por_grupo[otro].std(ddof=1), 1)}, en '
               f'{sum(por_grupo[g] > por_grupo[otro])} de las {len(runs)} '
               f'semillas (Wilcoxon de rangos con signo, $p$ = {_fmt_p(p)}).')
        return txt, {'aporte_grupo': g,
                     'aporte_pct': round(float(por_grupo[g].mean()), 2),
                     'aporte_p': p}

    res = compare_indicator(lambda l, c: por_grupo[l], grupos, None)
    if res is None:
        return None, {}
    gr = homogeneous_groups(res, grupos, res['medians'], True)
    txt = (f'  Repitiendo el cálculo dentro de cada semilla, el aporte se '
           f'contrasta con Friedman sobre las {len(runs)} semillas como '
           f'bloques ($p$ = {_fmt_p(res["p_omnibus"])}) y comparaciones por '
           f'pares con Wilcoxon corregidas por Holm.  Grupos homogéneos, de '
           f'mayor a menor aporte: {fmt_groups(gr)}.')
    return txt, {'aporte_grupo': ', '.join(gr[0]),
                 'aporte_pct': round(float(np.mean(por_grupo[gr[0][0]])), 2),
                 'aporte_p': res['p_omnibus']}


def _partir_etiqueta(nombre):
    """'NSGA-II (PCX)' → ('NSGA-II', 'PCX').  Sin paréntesis, el segundo campo
    queda vacío: es el caso de CMOPSO, que no tiene operadores."""
    if nombre.endswith(')') and '(' in nombre:
        alg, cruce = nombre.rsplit('(', 1)
        return alg.strip(), cruce[:-1].strip()
    return nombre, '---'


def write_contribucion_table(series, pf_df, nombre, out_dir,
                             grupo_de=None, etiqueta='Operador', nota=''):
    """Tabla LaTeX de la contribución al frente no dominado conjunto.

    Complementa al test sobre el hipervolumen, que mide la extensión del frente
    y no la calidad de lo que contiene.  Acá se junta lo producido por todas las
    series, se recalcula la no-dominancia global y se mira quién aportó los
    supervivientes: es dominancia de Pareto sobre los dos objetivos, sin
    ponderaciones.  Fsp3 no participa de la dominancia —es el constraint— pero se
    reporta como columna: junto a QED y SA describe qué clase de molécula pone
    cada serie en el frente, y en particular cuánto margen le deja al umbral.

    Las etiquetas del tipo 'NSGA-II (PCX)' se parten en dos columnas, con el
    algoritmo en \\multirow: repetirlo en cada fila haría creer que son
    entidades distintas, cuando son dos ramas de la misma.

    Devuelve el resumen del contraste para el CSV de la etapa.
    """
    grupo_de = grupo_de or pc._familia
    filas, _ = pc.contribucion_agregada(series, pf_df, grupo_de)
    por_grupo, compartidas, runs = pc.contribucion_por_semilla(series, grupo_de)
    _, resumen = _test_aporte(por_grupo, runs)

    partidas = [_partir_etiqueta(DISPLAY.get(f['nombre'], f['nombre']))
                for f in filas]
    total = filas[0]['total'] if filas else 0
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Contribución al frente no dominado conjunto en '
        f'{_latex_escape(nombre)}.  Se unen las soluciones de las series '
        f'comparadas sobre las {len(runs)} semillas, se deduplica por SMILES y '
        f'se recalcula la no-dominancia global ({total} soluciones).  «Aporta» '
        f'cuenta toda molécula del frente hallada por esa serie, por lo que las '
        f'compartidas suman en cada fila que las encontró; «exclusivas» solo '
        f'las que no halló ninguna otra.  Las tres últimas columnas son la media '
        f'sobre lo que cada serie aporta, y describen no cuánto sino qué aporta: '
        f'QED y SA son los objetivos, y Fsp3 va sin flecha porque es la '
        f'restricción ($\\geq$ {_num(pc.FSP3_MIN, 2)}) y no algo que se '
        f'optimice.{nota}}}',
        f'\\label{{tab:contribucion_{nombre.lower()}}}',
        r'\begin{tabular}{llrrrrrr}', r'\toprule',
        f'{etiqueta} & Operadores & Aporta & Exclusivas & \\% & QED $\\uparrow$ & '
        f'SA $\\downarrow$ & Fsp3 \\\\',
        r'\midrule',
    ]
    for i, (f, (alg, cruce)) in enumerate(zip(filas, partidas)):
        # Las filas de grupo, si las hay, van separadas: agregan sobre las
        # anteriores y no son sumables con ellas.
        if i == len(series) and len(filas) > len(series):
            lines.append(r'\midrule')
        # El nombre del algoritmo se escribe una sola vez, abarcando sus ramas.
        n_ramas = sum(1 for a, _ in partidas if a == alg)
        primera = i == 0 or partidas[i - 1][0] != alg
        if primera:
            if i:
                lines.append(r'\midrule')
            celda = (r'\multirow{%d}{*}{%s}' % (n_ramas, _latex_escape(alg))
                     if n_ramas > 1 else _latex_escape(alg))
        else:
            celda = ''
        lines.append(
            f"{celda} & {_latex_escape(cruce)} & "
            f"{f['aporta']} & {f['exclusiva']} & {_num(100*f['frac'], 1)} & "
            f"{_num(f['qed'], 3)} & {_num(f['sa'], 2)} & {_num(f['fsp3'], 3)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, f'contribucion_{nombre}.tex'))

    pd.DataFrame(filas).to_csv(
        os.path.join(out_dir, f'contribucion_{nombre}.csv'), index=False)

    resumen_filas = filas[len(series):] or filas
    for f in resumen_filas:
        print(f"  aporte {f['nombre']:>8s}: {f['aporta']:4d}/{total} "
              f"({100*f['frac']:4.1f}%)  excl. {f['exclusiva']:4d}  "
              f"QED<0.60 {100*f['qed_bajo']:4.1f}%  "
              f"Fsp3 en el borde {100*f['fsp3_borde']:4.1f}%")
    if por_grupo:
        detalle_txt = '   '.join(
            f'{g} {v.mean():.1f}%±{v.std(ddof=1):.1f}' for g, v in por_grupo.items())
        print(f"  por semilla: {detalle_txt}   compartidas {compartidas.mean():.1f}%")
    return resumen


def analyze_operators(alg, winners_dir, out_root, decision_col):
    # COMBO_DIRS fija el orden en que los combos aparecen en tablas y leyendas.
    series = pc.build_operator_series_winners(alg, winners_dir, COMBO_DIRS)
    if len(series) < 2:
        print(f"\n  ⚠ {alg}: {len(series)} combo(s) con datos; se omite")
        return None

    labels = [s.label for s in series]
    out_dir = os.path.join(out_root, alg)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'─'*64}\n  {alg}   combos: {', '.join(labels)}\n{'─'*64}")

    # Frente de referencia común a los 4 combos → IGD+ y ε+ comparables.
    pf_F, pf_df = pc.build_reference_front(series)
    indicator_data = {}
    if pf_F is not None:
        print(f"  frente de referencia: {len(pf_F)} soluciones no dominadas")
        indicator_data = pc.compute_indicators_per_run(series, pf_F)
        pf_df.to_csv(os.path.join(out_dir, f'reference_front_{alg}.csv'), index=False)

    get_values = pc._build_series_value_getter(series, indicator_data)

    # Las salidas de la sección 3.2: tablas de indicadores, frentes por combo y
    # la atribución del frente conjunto (tabla + figura).
    pc.generate_latex_comparison_tables(series, alg, out_dir, get_values)
    # Los frentes de cada combo por separado no se dibujan acá: en un mismo panel
    # se tapan entre sí, y la pregunta de esta etapa —quién sobrevive al juntarlos—
    # la responde mejor el frente conjunto de más abajo.
    for modo in pc.GRID_COLOR_MODES:
        pc.plot_pareto_qed_sa_grid(series, alg, out_dir, color_by=modo)

    if pf_df is not None:
        # La tabla de contribución por combo no se emite en esta etapa: lo que
        # decide está en las dos tablas de operadores de más abajo.  La figura
        # del frente conjunto sí, que es la que muestra el mecanismo.
        pc.plot_frente_conjunto(series, alg, out_dir, pf_df)

    # La comparación de operadores, sobre el indicador de decisión.
    label, higher = dict((c, (l, h)) for c, l, h in OP_INDICATORS)[decision_col]
    res = compare_indicator(get_values, labels, decision_col)
    if res is None:
        print(f"  ⚠ sin datos de {decision_col}; se omite el test")
        return None

    groups = homogeneous_groups(res, labels, res['medians'], higher)
    n_sig = sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)
    print(f"  Friedman ({label}): p = {res['p_omnibus']:.4g}   "
          f"({n_sig}/{len(res['pairs'])} pares significativos)")
    campeon = write_tabla_operadores(res, groups, labels, alg, out_dir,
                                     get_values, decision_col)

    return {'algorithm': alg, 'campeon': campeon,
            'p_friedman': res['p_omnibus'],
            'n_pares_sig': n_sig, 'n_pares': len(res['pairs']),
            'grupos': ' > '.join('{' + ', '.join(g) + '}' for g in groups),
            'mejor_grupo': ', '.join(groups[0]),
            'mediana': round(res['medians'][campeon], 6)}


def etapa2(args):
    algs = args.algorithms or GA_ALGS
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*64}")
    print("  ETAPA 2 — COMPARACIÓN DE OPERADORES")
    print(f"  Datos: {args.winners}")
    print(f"  Decisión por: {args.metric}")
    print(f"{'='*64}")

    rows = []
    for alg in algs:
        r = analyze_operators(alg, args.winners, args.out, args.metric)
        if r:
            rows.append(r)

    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(args.out, 'resumen_tests.csv')
        df.to_csv(path, index=False)
        print(f"\n{'='*64}\n  RESUMEN\n{'='*64}")
        print(df.to_string(index=False))
        print(f"\n  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 3 — comparación estadística final entre algoritmos
#
#   Lee finalistas/<ALG>/run_XX/ (la configuración elegida de cada algoritmo
#   tras las etapas 1 y 2) y contrasta los cinco entre sí sobre ALG_METRIC.
# ═══════════════════════════════════════════════════════════════════════════

ALG_METRIC = 'hypervolume'
ALG_METRIC_LABEL = 'hipervolumen'
ALG_HIGHER_BETTER = True


def write_groups_table(res, groups, out_dir, get_values, labels):
    """Las 10 comparaciones por pares, con el resultado resumido en grupos.

    Misma estructura que la tabla de operadores: al $p$ lo acompaña el tamaño de
    efecto, y el ganador va en su propia columna.  Donde el post-hoc no separa no
    se declara ganador, aunque las medianas ordenen."""
    vals = {l: np.asarray(get_values(l, ALG_METRIC), dtype=float) for l in labels}
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparación entre algoritmos sobre {ALG_METRIC_LABEL}.  '
        f'Test de Friedman con las 20 semillas como bloques '
        f'($p$ = {_fmt_p(res["p_omnibus"])}), seguido de las comparaciones por '
        f'pares con Wilcoxon de rangos con signo y corrección de Holm '
        f'($\\alpha = {_num(0.05, 2)}$); $r_{{rb}}$ es la correlación '
        f'rango-biserial de pares emparejados, con signo positivo cuando gana el '
        f'primero del par.  Grupos homogéneos, de mejor a peor: '
        f'{fmt_groups(groups)}.}}',
        r'\label{tab:comparacion_grupos}',
        r'\begin{tabular}{lrrl}', r'\toprule',
        r'Par & $p$ (Holm) & $r_{rb}$ & Mejor \\', r'\midrule',
    ]
    for p in res['pairs']:
        a, b = p['a'], p['b']
        r = rank_biserial(vals[a], vals[b])
        mejor = (DISPLAY.get(a if res['medians'][a] > res['medians'][b] else b,
                             a if res['medians'][a] > res['medians'][b] else b)
                 if p['p_holm'] < 0.05 else '---')
        lines.append(
            f"{DISPLAY.get(a, a)} vs {DISPLAY.get(b, b)} & "
            f"{_fmt_p(p['p_holm'])} & "
            f"{('$+$' if r >= 0 else '$-$') + _num(abs(r), 3)} & {mejor} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    path = os.path.join(out_dir, 'grupos_homogeneos.tex')
    print()
    _write_tex(lines, path, msg=path)


def etapa3(args):
    algs = [a for a in pc.ALGORITHM_ORDER
            if pc._has_runs(os.path.join(args.finalistas, a))]
    series = pc.build_finalist_series(algs, args.finalistas)
    if len(series) < 3:
        print(f"Se necesitan ≥3 algoritmos en {args.finalistas}")
        return
    labels = [s.label for s in series]
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*70}")
    print("  ETAPA 3 — COMPARACIÓN ENTRE ALGORITMOS")
    print(f"  {', '.join(DISPLAY.get(l, l) for l in labels)}")
    print(f"{'='*70}\n")

    pf, pf_df = pc.build_reference_front(series)
    ind = pc.compute_indicators_per_run(series, pf) if pf is not None else {}
    print(f"  frente de referencia: {len(pf) if pf is not None else 0} soluciones\n")
    get = pc._build_series_value_getter(series, ind)

    res = compare_indicator(get, labels, ALG_METRIC)
    if res is None:
        print(f"  ⚠ sin datos de {ALG_METRIC}")
        return
    groups = homogeneous_groups(res, labels, res['medians'], ALG_HIGHER_BETTER)
    n_sig = sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)

    print(f"  Friedman: p = {res['p_omnibus']:.3g}")
    print(f"  pares significativos tras Holm: {n_sig} de {len(res['pairs'])}\n")
    for lab in sorted(labels, key=lambda l: -res['medians'][l]):
        print(f"    {DISPLAY.get(lab, lab):10s} mediana = {res['medians'][lab]:.5f}")
    print("\n  grupos: " + ' > '.join('{' + ', '.join(g) + '}' for g in groups))

    write_groups_table(res, groups, args.out, get, labels)
    pd.DataFrame([{'a': p['a'], 'b': p['b'], 'p_raw': p['p_raw'],
                   'p_holm': p['p_holm'], 'significativo': p['p_holm'] < 0.05}
                  for p in res['pairs']]).to_csv(
        os.path.join(args.out, 'tests_pares.csv'), index=False)
    print(f"  ✓ {os.path.join(args.out, 'tests_pares.csv')}")

    analisis_frente_conjunto(args)


# ═══════════════════════════════════════════════════════════════════════════
#   Frente conjunto — el pool de candidatos
#
#   Va aparte de la etapa 3 porque responde otra pregunta.  La etapa 3 compara
#   algoritmos a igual presupuesto, una configuración cada uno; acá se juntan
#   las DOS familias de cruce de cada algoritmo, que es el material que llega a
#   la fase de afinidad, y se mira qué sobrevive al enfrentarlas y de dónde sale.
#
#   Como cada AG aporta dos configuraciones y CMOPSO una, los presupuestos no son
#   comparables: esto caracteriza el pool, no ordena algoritmos.
# ═══════════════════════════════════════════════════════════════════════════

# Cada familia de cruce aporta una rama al pool.
POOL_FAMILIAS = ['pcx', 'sbx']


def combos_pool(alg, winners_dir, alpha=0.05):
    """Los dos combos con que un algoritmo genético entra al pool: uno por
    familia de cruce y, dentro de cada familia, la mutación que gana.

    Se aplica la misma regla con que la etapa 2 elige su campeón —la ganadora si
    el post-hoc la separa, la estándar si no— pero por familia, y leyendo el
    mismo $p$ corregido por Holm que publica la tabla de esa etapa, para que la
    decisión se pueda seguir desde el cuadro.  Decide el hipervolumen, como en el
    resto del pipeline.

    En la práctica esto deja la mutación polinomial en todas las ramas salvo la
    SBX de MOEA/D y de AGE-MOEA, donde el test sí separa a las dos mutaciones
    (p = 0.038 y 0.021) y gana la gaussiana.  Fijar pm también ahí metía al pool
    la peor de las dos variantes SBX según el test; medido, el cambio no importa
    la cola de bajo QED y sube el aporte de esas dos ramas.

    Devuelve [(familia, combo)] en el orden de POOL_FAMILIAS."""
    series = pc.build_operator_series_winners(alg, winners_dir, COMBO_DIRS)
    if not series:
        return []          # CMOPSO: sin operadores no hay ramas que elegir
    hv = {s.label: pc.load_metrics(s.pop_dir).sort_values('run')['hypervolume'].values
          for s in series}
    res = compare_indicator(lambda lab, _col: hv[lab], list(hv), None)

    elegidos = []
    for fam in POOL_FAMILIAS:
        pm, gauss = f'{fam}_{MUT_STD}', f'{fam}_gauss'
        if pm not in hv:
            continue
        if gauss not in hv or res is None:
            elegidos.append((fam, pm))
            continue
        p = next((x['p_holm'] for x in res['pairs']
                  if {x['a'], x['b']} == {pm, gauss}), 1.0)
        gana_gauss = p < alpha and res['medians'][gauss] > res['medians'][pm]
        elegidos.append((fam, gauss if gana_gauss else pm))
    return elegidos


def _series_pool(winners_dir, finalistas_dir):
    """Las dos ramas de cruce de cada AG, más CMOPSO, que no tiene operadores."""
    series = []
    for alg in GA_ALGS:
        # El combo va en la etiqueta y no solo la familia: desde que la mutación
        # puede cambiar entre ramas, decir 'SBX' a secas escondería cuál es.
        for _, combo in combos_pool(alg, winners_dir):
            cfg_dir = pc.winner_cfg_dir(winners_dir, alg, combo)
            if cfg_dir:
                series.append(pc.Series(f'{DISPLAY.get(alg, alg)} ({combo})',
                                        cfg_dir))
    d = os.path.join(finalistas_dir, PSO_ALG)
    if pc._has_runs(d):
        series.append(pc.Series(PSO_ALG, d))
    return series


def _nota_pool(winners_dir):
    """Frase para el caption de la tabla del pool: con qué configuración entró
    cada algoritmo, y a dónde ir por la justificación.

    Las excepciones se arman con la misma regla que elige las ramas, así que la
    frase no puede quedar diciendo una cosa mientras el pool hace otra.  El
    detalle estadístico no se repite acá —vive en las tablas de la etapa 2— y se
    remite a ellas por \\ref: la sección del frente conjunto describe el material,
    no vuelve a discutir la comparación de operadores."""
    excepciones = []
    for alg in GA_ALGS:
        for fam, combo in combos_pool(alg, winners_dir):
            mut = combo.split('_')[1]
            if mut != MUT_STD:
                excepciones.append(
                    f'{_latex_escape(DISPLAY.get(alg, alg))} en {fam.upper()} '
                    f'({"gaussiana" if mut == "gauss" else _latex_escape(mut)}, '
                    f'cuadro~\\ref{{tab:ops_{alg.lower()}}})')

    nota = ('  Cada algoritmo genético entra con dos ramas, una por familia de '
            'cruce, en la configuración que ganó su bloque en la selección de '
            'hiperparámetros.  La mutación es la polinomial')
    if not excepciones:
        return nota + ' en todas las ramas.'
    detalle = (' y '.join(excepciones) if len(excepciones) < 3
               else ', '.join(excepciones[:-1]) + ' y ' + excepciones[-1])
    return (nota + ', salvo en las ramas donde la comparación de operadores '
            f'separó a las dos mutaciones: {detalle}.')


def _algoritmo_pool(label):
    """Agrupa por algoritmo: las dos ramas de cruce de un AG caen en el mismo
    grupo ('NSGA-II (PCX)' → 'NSGA-II'), y CMOPSO, que no tiene operadores, queda
    como el suyo.

    Es la agrupación de las figuras del frente conjunto.  Antes agrupaban por
    familia de cruce, que responde otra pregunta —de qué operador sale cada
    región— y ya la contesta la figura por algoritmo de la etapa 2.  Acá lo que
    interesa es qué algoritmo puso cada molécula del pool de candidatos."""
    return _partir_etiqueta(label)[0]


def analisis_frente_conjunto(args):
    series = _series_pool(args.winners, args.finalistas)
    if len(series) < 2:
        print(f"\n  ⚠ sin datos suficientes en {args.winners}; se omite el "
              f"frente conjunto")
        return

    print(f"\n{'='*70}")
    print("  FRENTE CONJUNTO — pool de candidatos")
    print(f"  {len(series)} configuraciones: {', '.join(s.label for s in series)}")
    print(f"{'='*70}\n")

    pf_F, pf_df = pc.build_reference_front(series)
    if pf_df is None:
        print("  ⚠ no se pudo construir el frente conjunto")
        return
    os.makedirs(args.out_frente, exist_ok=True)

    # El frente en sí, con la atribución: son las moléculas candidatas que pasan
    # a la fase de afinidad, así que conviene tenerlas y no solo su resumen.
    at = pc.atribuir_frente(series, pf_df, pc._por_serie)
    at.to_csv(os.path.join(args.out_frente, 'frente_pool.csv'), index=False)
    print(f"  ✓ frente_pool.csv  ({len(at)} moléculas)")

    # La tabla desglosa por configuración —es la pregunta de cuánto aporta cada
    # rama—; la figura agrupa por algoritmo, porque nueve colores serían
    # ilegibles y las dos ramas de un mismo AG no son entidades distintas.
    #
    # Ya no hay versión 3D: existía para mostrar la superficie del frente en
    # QED-SA-Fsp3, y con Fsp3 como constraint el frente es una curva.  Su lugar
    # lo ocupa el segundo panel de plot_frente_conjunto, que muestra dónde quedó
    # cada molécula respecto del umbral.
    write_contribucion_table(
        series, pf_df, 'pool', args.out_frente,
        grupo_de=pc._por_serie, etiqueta='Configuración',
        nota=_nota_pool(args.winners))
    pc.plot_frente_conjunto(series, 'pool', args.out_frente, pf_df,
                            grupo_de=_algoritmo_pool)


# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 4 — baselines contra los algoritmos multiobjetivo
#
#   Compara las cuatro baselines (cribado de MOSES, aleatorio, escalador, GA de
#   suma ponderada) con los cinco MOEAs ya seleccionados, sobre el mismo
#   presupuesto de 100.000 evaluaciones y las mismas 20 semillas.
# ═══════════════════════════════════════════════════════════════════════════

# Orden de peor a mejor esperado.
BASELINE_KEYS = ['WEIGHTED_GA', 'HILL_CLIMBER', 'RANDOM', 'SCREENING']


def _series_baselines(finalistas, baselines):
    """Los cinco MOEAs y las tres baselines como series comparables."""
    series = []
    for alg in pc.ALGORITHM_ORDER:
        d = os.path.join(finalistas, alg)
        if pc._has_runs(d):
            series.append(pc.Series(alg, d, color_key=alg))
    for m in BASELINE_KEYS:
        # <baselines>/<METHOD>/[tag/]pop{P}_gen{G}/
        for d in sorted(glob.glob(os.path.join(baselines, m, '*', '*')) +
                        glob.glob(os.path.join(baselines, m, '*'))):
            if pc._has_runs(d):
                series.append(pc.Series(m, d, color_key=m))
                break
    return series


def write_baseline_tables(res, groups, medians, stds, labels, metric_label,
                          out_dir):
    n_moea = sum(1 for l in labels if l not in BASELINE_KEYS)

    # Tabla 1: descriptiva
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Algoritmos multiobjetivo frente a las baselines sobre '
        f'{metric_label}, con idéntico presupuesto de 100.000 evaluaciones y las '
        f'mismas 20 semillas.  Mediana y desvío estándar entre ejecuciones.}}',
        r'\label{tab:baselines_desc}',
        r'\begin{tabular}{llc}', r'\toprule',
        r'& Método & ' + metric_label.capitalize() + r' \\', r'\midrule',
    ]
    for i, lab in enumerate(labels):
        if i == n_moea:
            lines.append(r'\midrule')
        tipo = ''
        if i == 0:
            tipo = r'\multirow{%d}{*}{MOEA}' % n_moea
        elif i == n_moea:
            tipo = r'\multirow{%d}{*}{Baseline}' % (len(labels) - n_moea)
        lines.append(f"{tipo} & {DISPLAY.get(lab, lab)} & "
                     f"${_num(medians[lab], 4)} \\pm {_num(stds[lab], 4)}$ \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, 'comparacion.tex'))

    # Tabla 2: cada MOEA contra cada baseline
    moeas = [l for l in labels if l not in BASELINE_KEYS]
    bases = [l for l in labels if l in BASELINE_KEYS]
    pmap = {}
    for p in res['pairs']:
        pmap[(p['a'], p['b'])] = p['p_holm']
        pmap[(p['b'], p['a'])] = p['p_holm']

    lines = [
        r'\begin{table}[htbp]', r'\centering',
        f'\\caption{{Comparaciones por pares entre cada algoritmo multiobjetivo y '
        f'cada baseline sobre {metric_label}.  Test de Friedman global '
        f'($p$ = {_fmt_p(res["p_omnibus"])}) y Wilcoxon de rangos con signo por '
        f'pares con corrección de Holm; se indica el $p$ ajustado.  '
        f'Grupos homogéneos: {fmt_groups(groups)}.}}',
        r'\label{tab:baselines_tests}',
        r'\begin{tabular}{l' + 'c' * len(bases) + '}', r'\toprule',
        'Algoritmo & ' + ' & '.join(DISPLAY.get(b, b) for b in bases) + r' \\',
        r'\midrule',
    ]
    for m in moeas:
        cells = [_fmt_p(pmap.get((m, b))) for b in bases]
        lines.append(f'{DISPLAY.get(m, m)} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    _write_tex(lines, os.path.join(out_dir, 'grupos_homogeneos.tex'))


def etapa4(args):
    series = _series_baselines(args.finalistas, args.baselines)
    labels = [s.label for s in series]
    faltan = [m for m in BASELINE_KEYS if m not in labels]
    if faltan:
        print(f"  ⚠ sin datos de: {', '.join(faltan)}")
    if len(series) < 3:
        print("Se necesitan más series")
        return
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*70}")
    print("  ETAPA 4 — BASELINES vs ALGORITMOS MULTIOBJETIVO")
    print(f"  {', '.join(DISPLAY.get(l, l) for l in labels)}")
    print(f"{'='*70}\n")

    pf, _ = pc.build_reference_front(series)
    ind = pc.compute_indicators_per_run(series, pf) if pf is not None else {}
    get = pc._build_series_value_getter(series, ind)

    res = compare_indicator(get, labels, args.metric)
    if res is None:
        print(f"  ⚠ sin datos de {args.metric}")
        return
    stds = {l: float(np.std(np.asarray(get(l, args.metric), float), ddof=1))
            for l in labels}
    groups = homogeneous_groups(res, labels, res['medians'], True)

    print(f"  Friedman: p = {res['p_omnibus']:.3g}   "
          f"({sum(1 for p in res['pairs'] if p['p_holm'] < 0.05)} de "
          f"{len(res['pairs'])} pares significativos)\n")
    for l in sorted(labels, key=lambda x: -res['medians'][x]):
        tipo = 'baseline' if l in BASELINE_KEYS else 'MOEA'
        print(f"    {DISPLAY.get(l, l):14s} {res['medians'][l]:.5f}  ({tipo})")
    print("\n  grupos: " + ' > '.join('{' + ', '.join(g) + '}' for g in groups))

    write_baseline_tables(res, groups, res['medians'], stds, labels,
                          args.metric_label, args.out)
    pd.DataFrame([{'a': p['a'], 'b': p['b'], 'p_raw': p['p_raw'],
                   'p_holm': p['p_holm'], 'significativo': p['p_holm'] < 0.05}
                  for p in res['pairs']]).to_csv(
        os.path.join(args.out, 'tests_pares.csv'), index=False)
    print("  ✓ tests_pares.csv")


# ═══════════════════════════════════════════════════════════════════════════
#   Moléculas representativas
#
#   Una imagen con las moléculas de mayor QED del frente de cada algoritmo.  El
#   frente se arma juntando las moléculas de las 20 ejecuciones de sus DOS ramas
#   de cruce —las mismas configuraciones del pool—, deduplicando por SMILES y
#   recalculando la dominancia global.
#
#   Va con el frente conjunto porque ilustra el material que llega a la fase de
#   afinidad, no las configuraciones que se compararon en la etapa 3.
# ═══════════════════════════════════════════════════════════════════════════

MOLECULAS_OUT = os.path.join(OUT_FRENTE, "moleculas_representativas.png")

N_MOLECULAS = 5      # por algoritmo

# Ventana de interés farmacológico.  Ya no lleva banda de Fsp3: el constraint la
# garantiza por construcción —molecules.csv solo publica moléculas factibles— y
# fijar un rango por encima del umbral seleccionaría a mano justo la cola que la
# búsqueda no tenía por qué producir, porque nada empuja Fsp3 más allá de 0.3.
# Queda el corte de SA, que sí desempata entre las decenas de moléculas
# empatadas en el QED máximo.
SA_MAX = 3.0


def load_front(alg, winners_dir, finalistas_dir):
    """Frente no dominado de un algoritmo sobre las configuraciones del pool.

    Para los AG son sus dos ramas de cruce juntas; CMOPSO no tiene operadores y
    va con su única configuración."""
    dfs = []
    for _, combo in combos_pool(alg, winners_dir):
        cfg_dir = pc.winner_cfg_dir(winners_dir, alg, combo)
        if cfg_dir:
            dfs.append(pc.load_pareto_molecules(cfg_dir))
    if not dfs:
        d = os.path.join(finalistas_dir, alg)
        if pc._has_runs(d):
            dfs.append(pc.load_pareto_molecules(d))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    return pc._compute_non_dominated(df.drop_duplicates(subset='smiles'))


def pick(front, n=N_MOLECULAS):
    """Las n moléculas de mayor QED con SA por debajo de SA_MAX.

    Ordenar solo por QED no sirve acá: en el frente hay decenas de moléculas
    empatadas en QED ≈ 0.948, así que manda el desempate.  El corte por SA se
    queda con las sintetizables de ese empate.

    La banda de Fsp3 que llevaba la etapa a 3 objetivos se retiró: ahí filtraba
    aromáticos planos que el frente sí contenía, y acá el constraint ya los dejó
    afuera antes de que llegaran a molecules.csv.

    Si el corte dejara el frente vacío se cae al frente completo, para que la
    figura se genere igual.
    """
    dentro = front[front['sa'] < SA_MAX]
    if dentro.empty:
        dentro = front
    return dentro.nlargest(n, 'qed').reset_index(drop=True)


def render(smiles, size=(420, 320)):
    """PNG de la estructura, como array para matplotlib."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = d.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return mpimg.imread(io.BytesIO(d.GetDrawingText()), format='png')


def moleculas(args):
    algs = [a for a in pc.ALGORITHM_ORDER
            if pc._has_runs(os.path.join(args.finalistas, a))]
    fronts = {a: load_front(a, args.winners, args.finalistas) for a in algs}
    fronts = {a: f for a, f in fronts.items() if not f.empty}
    if not fronts:
        print(f"No se encontraron frentes en {args.finalistas}")
        return

    seleccion = {a: pick(f) for a, f in fronts.items()}

    nrows, ncols = len(fronts), N_MOLECULAS
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    for i, (alg, sel) in enumerate(seleccion.items()):
        for j in range(ncols):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#cccccc')
            if j >= len(sel):
                ax.set_visible(False)
                continue

            m = sel.iloc[j]
            img = render(m['smiles'])
            if img is not None:
                ax.imshow(img)
            # Los dos objetivos más Fsp3: no seleccionó nada, pero deja ver con
            # cuánto margen sobre el umbral quedó cada estructura dibujada.
            ax.set_xlabel(f"QED {m['qed']:.3f}  ·  SA {m['sa']:.2f}  "
                          f"·  Fsp3 {m['fsp3']:.2f}",
                          fontsize=9, labelpad=3)
            if j == 0:
                ax.set_ylabel(DISPLAY.get(alg, alg), fontsize=13,
                              fontweight='bold', labelpad=10)

    fig.suptitle(f'Las {N_MOLECULAS} moléculas de mayor QED del frente de cada '
                 f'algoritmo, con SA < {SA_MAX:g} '
                 f'(todas cumplen Fsp3 ≥ {pc.FSP3_MIN:g})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ {args.out}")

    # Detalle de las moléculas elegidas
    rows = []
    for alg, sel in seleccion.items():
        for j, m in sel.iterrows():
            rows.append({'algoritmo': DISPLAY.get(alg, alg), 'puesto': j + 1,
                         'qed': round(m['qed'], 4), 'sa': round(m['sa'], 2),
                         'fsp3': m['fsp3'], 'smiles': m['smiles']})
    out = pd.DataFrame(rows)
    csv = os.path.splitext(args.out)[0] + '.csv'
    out.to_csv(csv, index=False)
    print(f"✓ {csv}\n")
    print(out.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
#   CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Análisis de los experimentos multiobjetivo.")
    sub = ap.add_subparsers(dest='etapa', required=True,
                            metavar='etapa1|etapa2|etapa3|etapa4|moleculas')
    fmt = argparse.ArgumentDefaultsHelpFormatter

    p1 = sub.add_parser('etapa1', formatter_class=fmt,
                        help="Selección de hiperparámetros por combinación de "
                             "operadores.")
    p1.add_argument('--csv', default=METRICS_CSV, help="CSV consolidado del grid.")
    p1.add_argument('--out', default=OUT_HP, help="Directorio de salida.")
    p1.add_argument('--algorithms', nargs='+', default=None,
                    help="Algoritmos a analizar (default: todos).")
    p1.add_argument('--metric', default='hypervolume', choices=list(HP_METRICS),
                    help="Métrica de selección.")
    p1.set_defaults(func=etapa1)

    p2 = sub.add_parser('etapa2', formatter_class=fmt,
                        help="Comparación de operadores por algoritmo.")
    p2.add_argument('--winners', default=WINNERS_DIR)
    p2.add_argument('--out', default=OUT_OPERADORES)
    p2.add_argument('--algorithms', nargs='+', default=None)
    p2.add_argument('--metric', default='hypervolume',
                    choices=[c for c, _, _ in OP_INDICATORS],
                    help="Indicador con el que se elige el combo ganador.")
    p2.set_defaults(func=etapa2)

    p3 = sub.add_parser('etapa3', formatter_class=fmt,
                        help="Comparación estadística entre algoritmos.")
    p3.add_argument('--finalistas', default=FINALISTAS_DIR)
    p3.add_argument('--out', default=OUT_ALGORITMOS)
    p3.add_argument('--out-frente', default=OUT_FRENTE,
                    help="Directorio del análisis del frente conjunto.")
    p3.add_argument('--winners', default=WINNERS_DIR,
                    help="De acá salen las dos ramas de cruce del pool.")
    p3.set_defaults(func=etapa3)

    p4 = sub.add_parser('etapa4', formatter_class=fmt,
                        help="Baselines vs algoritmos multiobjetivo.")
    p4.add_argument('--finalistas', default=FINALISTAS_DIR)
    p4.add_argument('--baselines', default=BASELINES_DIR)
    p4.add_argument('--out', default=OUT_BASELINES)
    p4.add_argument('--metric', default='hypervolume')
    p4.add_argument('--metric-label', default='hipervolumen')
    p4.set_defaults(func=etapa4)

    pm = sub.add_parser('moleculas', formatter_class=fmt,
                        help="Moléculas representativas del frente de cada "
                             "algoritmo.")
    pm.add_argument('--finalistas', default=FINALISTAS_DIR)
    pm.add_argument('--winners', default=WINNERS_DIR,
                    help="De acá salen las dos ramas de cruce de cada algoritmo.")
    pm.add_argument('--out', default=MOLECULAS_OUT)
    pm.set_defaults(func=moleculas)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
