"""
PDF con una tabla por algoritmo: sus combinaciones de operadores y, dentro de
cada una, las tres configuraciones de mutación.

En cada métrica va en negrita la mejor de las tres mutaciones de esa combinación.
CMOPSO no tiene operadores, así que su tabla lleva una sola combinación.

Con --con-main se agregan, del grid de main, las tres mutaciones anteriores con
todo lo demás igual: mismo algoritmo, combinación, reparto y probabilidad de cruce
(en CMOPSO, misma élite y velocidad).  Así lo único que cambia entre las seis filas
de cada combinación es la mutación.
"""

import os
import subprocess

import pandas as pd

from .comun import ROOT_DIR
from .etapa1 import (DISPLAY, MULTIOBJETIVO, ORDEN_ALG, agregar_indicadores,
                     cargar)


# Los encabezados que no existen en la fuente de LaTeX, o que en los PDFs van con
# otro nombre que en las tablas de la etapa 1.
TEX = {'ε+': r'$\epsilon^+$', 'IGD+': r'IGD$^+$', 'Fsp3': r'Fsp$_3$',
       'Tamaño de Pareto': 'Pareto size'}

# Las columnas de las tablas: las multiobjetivo de la etapa 1, más la validez, y el
# tiempo al final.
COLUMNAS = ([c for c in MULTIOBJETIVO if c[0] != 'time_sec']
            + [('validity', 'Validez', 4, '↑')]
            + [c for c in MULTIOBJETIVO if c[0] == 'time_sec'])


CONFIGS = ['0.05', '0.1', 'adaptativa']

# Las mutaciones del grid de main, que --con-main pone arriba de las de exp3.
CONFIGS_MAIN = ['0.004', '0.012', '0.031']

MAIN_GRID = os.path.join(ROOT_DIR, 'results', 'main', 'grid')

# Lo que tiene que coincidir entre una corrida de main y una de exp3 para que solo
# difieran en la mutación.  'tipo' y no 'mutation': en exp3 la adaptativa es pm_adapt.
CLAVES_GA  = ['algorithm', 'crossover', 'tipo', 'cx_prob', 'pop_size', 'n_gen']
CLAVES_PSO = ['algorithm', 'elite_size', 'vel_rate', 'pop_size', 'n_gen']

CABECERA = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish]{babel}
\usepackage[margin=1.5cm,a4paper,landscape]{geometry}
\usepackage{booktabs,amsmath,capt-of}
\renewcommand{\arraystretch}{1.2}
\pagestyle{empty}
\begin{document}
"""


def _num(v, dec):
    return f'{v:.{dec}f}'.replace('.', '{,}')


def _con_main(d):
    """Suma a las corridas de exp3 las del grid de main que se diferencian de
    ellas solo en la mutación."""
    m = cargar(MAIN_GRID)
    ga = d[d.algorithm != 'CMOPSO'][CLAVES_GA].drop_duplicates()
    pso = d[d.algorithm == 'CMOPSO'][CLAVES_PSO].drop_duplicates()
    m = pd.concat([m.merge(ga, on=CLAVES_GA), m.merge(pso, on=CLAVES_PSO)])
    print(f"  del grid de main: {len(m)} corridas con el mismo cruce y reparto")
    # índice nuevo: agregar_indicadores ubica cada corrida por índice
    return pd.concat([m, d], ignore_index=True)


def _fijo(g, alg):
    """Lo que es igual en todas las filas de la tabla, para el título."""
    if alg == 'CMOPSO':
        return (f'élite {g.elite_size.iloc[0]:g}, '
                f'$v_{{\\max}}$ = {_num(g.vel_rate.iloc[0], 1)}')
    cx = ' y '.join(f'{" / ".join(_num(v, 1) for v in sorted(g[g.cruce == c].cx_prob.unique()))} ({c})'
                    for c in ('PCX', 'SBX'))
    return f'$p_c$ = {cx}'


def _tabla(d, alg, columnas, reparto, configs=CONFIGS):
    g = d[d.algorithm == alg]
    cols = [(c, e, dec, fl) for c, e, dec, fl in columnas
            if c in g.columns and not g[c].isna().all()]
    combos = (['—'] if alg == 'CMOPSO'
              else ['PCX pm', 'PCX gauss', 'SBX pm', 'SBX gauss'])

    filas = []
    for combo in combos:
        if combo == '—':
            s0 = g
        else:
            cx, tp = combo.split()
            s0 = g[(g.cruce == cx) & (g.tipo == tp)]
        bloque = []
        for cfg in configs:
            s = s0[s0.config == cfg]
            if s.empty:
                continue
            bloque.append({'combo': combo, 'mut': cfg,
                           **{c: (s[c].mean(), s[c].std()) for c, *_ in cols}})
        # la mejor de las tres, por métrica
        for c, _, _, fl in cols:
            if not fl or not bloque:
                continue
            vals = [b[c][0] for b in bloque]
            i = vals.index(max(vals) if fl == '↑' else min(vals))
            bloque[i][f'_best_{c}'] = True
        filas += bloque

    enc = ' & '.join(r'\textbf{' + TEX.get(e, e) + '}'
                     + (r' $\uparrow$' if fl == '↑' else
                        r' $\downarrow$' if fl == '↓' else '')
                     for _, e, _, fl in cols)
    if configs == CONFIGS:
        titulo = (f'{DISPLAY[alg]} — reparto {reparto}.  Media y desvío sobre las 20 '
                  f'semillas; en negrita la mejor de las tres configuraciones de '
                  f'mutación dentro de cada combinación.')
    else:
        titulo = (f'{DISPLAY[alg]} — reparto {reparto}, {_fijo(g, alg)}.  Media y '
                  f'desvío sobre las 20 semillas.  En cada combinación, sobre la '
                  f'línea las mutaciones del estudio de hiperparámetros y debajo las '
                  f'del experimento de mutación; en negrita la mejor de las seis.')
    out = [r'\begin{center}',
           f'\\captionof{{table}}{{{titulo}}}',
           r'\begin{tabular}{ll' + 'c' * len(cols) + '}', r'\toprule',
           r'\textbf{Operadores} & \textbf{Mutación} & ' + enc + r' \\', r'\midrule']
    previo = mut_previo = None
    for f in filas:
        if previo is not None and f['combo'] != previo:
            out.append(r'\midrule')
        elif (f['combo'] == previo and f['mut'] not in CONFIGS_MAIN
              and mut_previo in CONFIGS_MAIN):
            out.append(r'\cmidrule(l){2-' + str(2 + len(cols)) + '}')
        f['_mut_previo'] = (f.get('mut') == mut_previo and f['combo'] == previo)
        mut_previo = f.get('mut')
        celdas = ['' if f['combo'] == previo else f['combo'].replace('—', '--'),
                  f['mut'].replace('adaptativa', 'adapt.')]
        for c, _, dec, _ in cols:
            m, s = f[c]
            txt = f'{_num(m, dec)} \\pm {_num(s, dec)}'
            celdas.append(f'$\\mathbf{{{txt}}}$' if f.get(f'_best_{c}')
                          else f'${txt}$')
        out.append(' & '.join(celdas) + r' \\')
        previo = f['combo']
    out += [r'\bottomrule', r'\end{tabular}', r'\end{center}']
    return '\n'.join(out)


def tablas_pdf(args):
    d = cargar(args.results)
    if args.con_main:
        d = _con_main(d)
    d = agregar_indicadores(d)
    d = d[d.reparto == args.reparto]
    configs = CONFIGS_MAIN + CONFIGS if args.con_main else CONFIGS
    if d.empty:
        raise SystemExit(f"no hay corridas con reparto {args.reparto}")
    os.makedirs(args.out, exist_ok=True)

    partes = []
    for alg in ORDEN_ALG:
        if not (d.algorithm == alg).any():
            continue
        partes.append(_tabla(d, alg, COLUMNAS, args.reparto, configs))
    cuerpo = '\n\\clearpage\n'.join(partes)
    base = os.path.join(args.out, f'mutacion_{args.reparto}'
                                  + ('_con_main' if args.con_main else ''))
    with open(base + '.tex', 'w') as fh:
        fh.write(CABECERA + cuerpo + '\n\\end{document}\n')

    for _ in range(2):
        p = subprocess.run(['pdflatex', '-interaction=nonstopmode',
                            '-output-directory', args.out, base + '.tex'],
                           capture_output=True, text=True)
    if os.path.exists(base + '.pdf'):
        for ext in ('.aux', '.log'):
            if os.path.exists(base + ext):
                os.remove(base + ext)
        print(f"  ✓ {base}.pdf")
    else:
        print('\n'.join(p.stdout.splitlines()[-12:]))


# ─── Comparación entre presupuestos ──────────────────────────────────────────

def _tabla_reparto(d, alg, columnas):
    """Una fila por combinación de operadores y presupuesto, con la mutación ya
    elegida.  En negrita el mejor de los dos presupuestos en cada métrica."""
    g = d[d.algorithm == alg]
    cols = [(c, e, dec, fl) for c, e, dec, fl in columnas
            if c in g.columns and not g[c].isna().all()]
    combos = (['—'] if alg == 'CMOPSO'
              else ['PCX pm', 'PCX gauss', 'SBX pm', 'SBX gauss'])
    repartos = sorted(g.reparto.unique(), key=lambda r: -int(r.split('x')[1]))

    filas = []
    for combo in combos:
        sub0 = g if combo == '—' else g[(g.cruce == combo.split()[0])
                                        & (g.tipo == combo.split()[1])]
        # Sin mutación fijada, cada una va en su propia fila: si no, la tabla
        # promedia las tres y no se ve de dónde sale la diferencia.
        muts = CONFIGS if MUT == 'todas' else [MUT]
        for mut in muts:
            sub = sub0 if MUT != 'todas' else sub0[sub0.config == mut]
            bloque = []
            for rep in repartos:
                s = sub[sub.reparto == rep]
                if s.empty:
                    continue
                bloque.append({'combo': combo, 'mut': mut, 'rep': rep,
                               **{c: (s[c].mean(), s[c].std()) for c, *_ in cols}})
            for c, _, _, fl in cols:
                if not fl or not bloque:
                    continue
                vals = [b[c][0] for b in bloque]
                i = vals.index(max(vals) if fl == '↑' else min(vals))
                bloque[i][f'_best_{c}'] = True
            filas += bloque

    enc = ' & '.join(r'\textbf{' + TEX.get(e, e) + '}'
                     + (r' $\uparrow$' if fl == '↑' else
                        r' $\downarrow$' if fl == '↓' else '')
                     for _, e, _, fl in cols)
    col_mut = MUT == 'todas'
    out = [r'\begin{center}', r'\small' if col_mut else '',
           f'\\captionof{{table}}{{{DISPLAY[alg]} — los dos presupuestos '
           + (f'con mutación {MUT}.  ' if MUT != 'todas' else
              'con cada una de las tres configuraciones de mutación.  ')
           + f'Media y desvío sobre las semillas; en negrita el mejor de los dos '
             f'dentro de cada combinación.}}',
           r'\begin{tabular}{' + ('lll' if col_mut else 'll') + 'c' * len(cols) + '}',
           r'\toprule',
           r'\textbf{Operadores} & '
           + (r'\textbf{Mut.} & ' if col_mut else '')
           + r'\textbf{Reparto} & ' + enc + r' \\', r'\midrule']
    previo = mut_previo = None
    for f in filas:
        if previo is not None and f['combo'] != previo:
            out.append(r'\midrule')
        f['_mut_previo'] = (f.get('mut') == mut_previo and f['combo'] == previo)
        mut_previo = f.get('mut')
        celdas = ['' if f['combo'] == previo else f['combo'].replace('—', '--')]
        if col_mut:
            celdas.append('' if f.get('_mut_previo') else
                          f['mut'].replace('adaptativa', 'adapt.'))
        celdas.append(f['rep'])
        for c, _, dec, _ in cols:
            m, s = f[c]
            txt = f'{_num(m, dec)} \\pm {_num(s, dec)}'
            celdas.append(f'$\\mathbf{{{txt}}}$' if f.get(f'_best_{c}') else f'${txt}$')
        out.append(' & '.join(celdas) + r' \\')
        previo = f['combo']
    out += [r'\bottomrule', r'\end{tabular}', r'\end{center}']
    return '\n'.join(out)


MUT = '0.1'


def comparar_repartos(args):
    d = agregar_indicadores(cargar(args.results))
    if args.mutacion != 'todas':
        d = d[d.config == args.mutacion]
        if d.empty:
            raise SystemExit(f"no hay corridas con mutación {args.mutacion}")
    globals()['MUT'] = args.mutacion
    os.makedirs(args.out, exist_ok=True)

    partes = [_tabla_reparto(d, alg, COLUMNAS)
              for alg in ORDEN_ALG if (d.algorithm == alg).any()]
    base = os.path.join(args.out, f'repartos_mut{args.mutacion}')
    with open(base + '.tex', 'w') as fh:
        fh.write(CABECERA + '\n\\clearpage\n'.join(partes) + '\n\\end{document}\n')
    for _ in range(2):
        p = subprocess.run(['pdflatex', '-interaction=nonstopmode',
                            '-output-directory', args.out, base + '.tex'],
                           capture_output=True, text=True)
    if os.path.exists(base + '.pdf'):
        for ext in ('.aux', '.log'):
            if os.path.exists(base + ext):
                os.remove(base + ext)
        print(f"  ✓ {base}.pdf")
    else:
        print('\n'.join(p.stdout.splitlines()[-12:]))
