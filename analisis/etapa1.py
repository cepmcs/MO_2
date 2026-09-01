"""
Etapa 1 — selección de hiperparámetros.

De las 513 configuraciones del grid elige 17: la mejor de cada combo de
operadores en los 4 AG, más la mejor global de CMOPSO.  Gana la de menor rango
medio de hipervolumen, rankeando dentro de cada semilla.

Los operadores no se testean acá; esa es la etapa 2.

Deja la figura de selección y selected_configs en .csv y .tex.  El CSV no es
solo para leer: train.sh lo parsea en el cluster para saber qué 17 directorios
meter en el tar que baja al PC, así que sus nombres de columna son contrato.
"""

import math
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from .comun import (
    ALGORITHM_ORDER,
    DISPLAY,
    GA_ALGS,
    PSO_ALG,
    SEP_DECIMAL,
    _latex_escape,
    _num,
    _write_tex,
)



# ═══════════════════════════════════════════════════════════════════════════
#   Etapa 1 — selección de hiperparámetros
#
#   De las 513 configuraciones del grid elige 17: la mejor de cada combo de
#   operadores en los 4 GA, más la mejor global de CMOPSO.  Gana la de menor rango
#   medio de hipervolumen, rankeando dentro de cada semilla.
#
#   Los operadores no se testean acá; esa es la etapa 2.
#
#   Deja la figura de selección y selected_configs en .csv y .tex.  El CSV lo
#   parsea train.sh para armar winners/ en el cluster (ver su paso 2).
# ═══════════════════════════════════════════════════════════════════════════

# CMOPSO no tiene operadores, así que su panel se colorea por una perilla propia.
# Va el tope de velocidad y no el archivo de elites: sobre las 81 configuraciones,
# vel_rate explica el 45% de la varianza del hipervolumen y el 83% de la de la
# validez, mientras que elite_size explica 0.4% y 0.02%.  Coloreado por elites el
# panel salía sin estructura —los tres tonos mezclados a lo largo de toda la
# columna—, que es justamente lo que la figura tiene que dejar ver.
# Rampa secuencial porque el factor es ordinal: más claro, menos velocidad.
VEL_COLORS = {0.1: '#9ECAE1', 0.2: '#4292C6', 0.35: '#08519C'}


# columna: (etiqueta, higher_better)
HP_METRICS = {
    'hypervolume': ('Hipervolumen', True),
    'spacing':     ('Espaciamiento', False),
    'n_pareto':    ('Tamaño de Pareto', True),
    'validity':    ('Validez', True),
    # Fracción de las válidas que cumplen el constraint: cuánto del presupuesto
    # se gasta fuera de la región admisible.
    'feasibility': ('Factibilidad', True),
    'novelty':     ('Novedad', True),
    'best_sa':     ('Mejor SA', False),
    'time_sec':    ('Tiempo (s)', False),
}


# 'budget' agrupa pop_size×n_gen: están acoplados por el presupuesto fijo de 100k
# evaluaciones.
FACTORS_GA = ['budget', 'crossover', 'mutation', 'cx_prob', 'mut_prob']

# CMOPSO barre el archivo de elites, la mutación por-gen y el tope de velocidad
# (ver run_experiments.py).
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
        # vel_rate llega del CSV como float; se redondea antes de indexar el
        # mapa de colores para no depender de la representación exacta con que
        # pandas parseó 0.35.
        key = pd.Index([round(float(v), 4)
                        for v in m.index.get_level_values('vel_rate')])
        presentes = set(key)
        groups = [(e, c) for e, c in sorted(VEL_COLORS.items())
                  if e in presentes]

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
                 else VEL_COLORS.get(round(float(lv.get('vel_rate', -1)), 4),
                                     '#333333'))
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
        bloques.append((f'{PSO_ALG}: tope de velocidad',
                        [punto(c, label=f'$v_{{\\max}}$ = {e:g}')
                         for e, c in sorted(VEL_COLORS.items())]))
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
    """Los cinco algoritmos en una imagen: un panel por algoritmo con su grid de
    configuraciones y la elegida de cada bloque resaltada.

    El eje de validez es común a los cinco paneles; el de la métrica es propio de
    cada uno, porque los rangos difieren y compartirlo los aplastaría."""
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

# Naranja y azul de Okabe-Ito (los mismos con que figuras.py separa PCX de
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

    Responde qué perilla importó, que es lo que la figura de selección no dice.
    El eje vertical es común para que las alturas sean comparables."""
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
    for alg in [a for a in ALGORITHM_ORDER if a in per_alg]:
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
        cfg = re.sub(r'(\d)\.(\d)', rf'\1{SEP_DECIMAL}\2',
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
    algs = args.algorithms or [a for a in ALGORITHM_ORDER
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
