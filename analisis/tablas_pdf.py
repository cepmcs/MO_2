"""
PDF con una tabla por algoritmo: sus combinaciones de operadores y, dentro de
cada una, las tres configuraciones de mutación.

En cada métrica va en negrita la mejor de las tres mutaciones de esa combinación.
CMOPSO no tiene operadores, así que su tabla lleva una sola combinación.
"""

import os
import subprocess

import pandas as pd

from .etapa1 import (DISPLAY, MULTIOBJETIVO, ORDEN_ALG, agregar_indicadores,
                     cargar, comparar)


# Los encabezados que no existen en la fuente de LaTeX.
TEX = {'ε+': r'$\epsilon^+$', 'IGD+': r'IGD$^+$', 'Fsp3': r'Fsp$_3$'}


CONFIGS = ['0.05', '0.1', 'adaptativa']

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


def _tabla(d, alg, columnas, reparto):
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
        for cfg in CONFIGS:
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
    out = [r'\begin{center}',
           f'\\captionof{{table}}{{{DISPLAY[alg]} — reparto {reparto}.  '
           f'Media y desvío sobre las 20 semillas; en negrita la mejor de las tres '
           f'configuraciones de mutación dentro de cada combinación.}}',
           r'\begin{tabular}{ll' + 'c' * len(cols) + '}', r'\toprule',
           r'\textbf{Operadores} & \textbf{Mutación} & ' + enc + r' \\', r'\midrule']
    previo = mut_previo = None
    for f in filas:
        if previo is not None and f['combo'] != previo:
            out.append(r'\midrule')
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


ABREV = {'0.05': '.05', '0.1': '.1', 'adaptativa': 'ad'}


def _grupos(d, alg, columnas):
    """Los grupos homogéneos de las tres mutaciones, uno por combinación de
    operadores.  Va apaisada —métricas en filas, combinaciones en columnas— para
    que entre en la misma página que la tabla de arriba.

    Un test por combinación y no uno agregado: la mejor mutación puede no ser la
    misma en todas.  Friedman con las 20 semillas como bloques y, cuando separa,
    Wilcoxon por pares con Holm.  Donde el test no separa va un guion: las tres
    quedan empatadas."""
    g = d[d.algorithm == alg]
    combos = (['—'] if alg == 'CMOPSO'
              else ['PCX pm', 'PCX gauss', 'SBX pm', 'SBX gauss'])
    metricas = [(c, e, fl) for c, e, _, fl in columnas
                if fl and c in g.columns and not g[c].isna().all()]

    celdas = {}
    for combo in combos:
        if combo == '—':
            sub = g
        else:
            cx, tp = combo.split()
            sub = g[(g.cruce == cx) & (g.tipo == tp)]
        for col, enc, fl in metricas:
            res = comparar(sub, CONFIGS, 'config', col, fl == '↑', ['run'])
            if res is None:
                celdas[(combo, col)] = '--'
            elif res['p_omnibus'] >= 0.05 or len(res['grupos']) == 1:
                celdas[(combo, col)] = '--'
            else:
                celdas[(combo, col)] = ' $>$ '.join(
                    '\\{' + ','.join(ABREV.get(x, x) for x in gr) + '\\}'
                    for gr in res['grupos'])

    out = [r'\vspace{0.6em}', r'\begin{center}', r'\footnotesize',
           r'\captionof{table}{Grupos homogéneos de las tres configuraciones de '
           r'mutación dentro de cada combinación de operadores.  Friedman con las '
           r'20 semillas como bloques y Wilcoxon por pares con corrección de Holm '
           r'($\alpha = 0{,}05$); un guion indica que el test no las separa.  '
           r'Abreviaturas: .05, .1 y ad (adaptativa).}',
           r'\begin{tabular}{l' + 'c' * len(combos) + '}', r'\toprule',
           r'\textbf{Métrica} & '
           + ' & '.join(r'\textbf{' + c.replace('—', '--') + '}' for c in combos)
           + r' \\', r'\midrule']
    for col, enc, fl in metricas:
        out.append(TEX.get(enc, enc) + ' & '
                   + ' & '.join(celdas[(c, col)] for c in combos) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{center}']
    return '\n'.join(out)


def tablas_pdf(args):
    d = agregar_indicadores(cargar(args.results))
    d = d[d.reparto == args.reparto]
    if d.empty:
        raise SystemExit(f"no hay corridas con reparto {args.reparto}")
    os.makedirs(args.out, exist_ok=True)

    partes = []
    for alg in ORDEN_ALG:
        if not (d.algorithm == alg).any():
            continue
        partes.append(_tabla(d, alg, MULTIOBJETIVO, args.reparto)
                      + '\n\n' + _grupos(d, alg, MULTIOBJETIVO))
    cuerpo = '\n\\clearpage\n'.join(partes)
    base = os.path.join(args.out, f'mutacion_{args.reparto}')
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


def _grupos_reparto(d, alg, columnas):
    """Wilcoxon pareado entre los dos presupuestos, por combinación.  Con dos
    niveles no hay grupos que armar: se indica cuál gana, o un guion si el test
    no los separa."""
    g = d[d.algorithm == alg]
    combos = (['—'] if alg == 'CMOPSO'
              else ['PCX pm', 'PCX gauss', 'SBX pm', 'SBX gauss'])
    repartos = sorted(g.reparto.unique(), key=lambda r: -int(r.split('x')[1]))
    metricas = [(c, e, fl) for c, e, _, fl in columnas
                if fl and c in g.columns and not g[c].isna().all()]

    celdas = {}
    for combo in combos:
        sub = g if combo == '—' else g[(g.cruce == combo.split()[0])
                                       & (g.tipo == combo.split()[1])]
        bloques = ['run'] if MUT != 'todas' else ['run', 'config']
        for col, enc, fl in metricas:
            res = comparar(sub, repartos, 'reparto', col, fl == '↑', bloques)
            celdas[(combo, col)] = ('--' if res is None or res['p_omnibus'] >= 0.05
                                    else res['grupos'][0][0])

    out = [r'\vspace{0.6em}', r'\begin{center}', r'\footnotesize',
           r'\captionof{table}{Presupuesto que gana dentro de cada combinación de '
           r'operadores (Wilcoxon de rangos con signo, $\alpha = 0{,}05$); un '
           r'guion indica que el test no los separa.  Los bloques son las 20 '
           + ('semillas.}' if MUT != 'todas' else
              r'semillas por las tres configuraciones de mutación.}'),
           r'\begin{tabular}{l' + 'c' * len(combos) + '}', r'\toprule',
           r'\textbf{Métrica} & '
           + ' & '.join(r'\textbf{' + c.replace('—', '--') + '}' for c in combos)
           + r' \\', r'\midrule']
    for col, enc, fl in metricas:
        out.append(TEX.get(enc, enc) + ' & '
                   + ' & '.join(celdas[(c, col)] for c in combos) + r' \\')
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

    partes = [_tabla_reparto(d, alg, MULTIOBJETIVO) + '\n\n'
              + _grupos_reparto(d, alg, MULTIOBJETIVO)
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
