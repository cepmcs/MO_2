"""
Los dos comandos del análisis de exp4.

Las seis configuraciones son tres algoritmos (NSGA-II, NSGA-III y AGE-MOEA) por dos
cruces (PCX y SBX).  Se analizan juntas:

  comparacion  las seis en una sola comparación: las tablas, la convergencia
               de los indicadores, los frentes de cada configuración y la
               distribución de sus propiedades.  Sin tests estadísticos (pocas corridas).
  candidatos   el frente conjunto de las seis, de dónde sale y las 15 de mejor
               docking.  Escribe además los frentes en CSV (carpeta frentes).

Salida, en --out: tablas/ y figuras/ numeradas en el orden en que se cuentan, y
frentes/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .comun import (
    FSP3_MIN,
    _num_es,
    load_metrics,
    load_pareto_molecules,
    series_pool,
)
from .indicadores import (
    _partir_etiqueta,
    _por_serie,
    atribuir_frente,
    build_reference_front,
    load_reference_front,
    write_contribucion_table,
)
from .figuras import (
    _combined_pareto_fronts,
    _generate_report,
    plot_frente_conjunto,
    render,
)


def _algoritmo(label):
    """'NSGA-II (PCX)' → 'NSGA-II'."""
    return _partir_etiqueta(label)[0]



def _cruce(label):
    """'NSGA-II (PCX)' → 'PCX'."""
    return _partir_etiqueta(label)[1]



def _mutacion_por_cruce(series):
    """{cruce: probabilidad de mutación} leída de los metrics.csv, o {} si algún
    cruce corrió con más de una."""
    m = pd.concat([load_metrics(s.pop_dir) for s in series], ignore_index=True)
    out = {}
    for cruce, g in m.groupby('crossover'):
        valores = sorted(set(g['mut_prob']))
        if len(valores) != 1:
            return {}
        out[cruce] = valores[0]
    return out



def _texto_mutacion(series):
    mut = _mutacion_por_cruce(series)
    return ', '.join(f'{c.upper()} con {_num_es(v, "g")}' for c, v in mut.items())



# ═══════════════════════════════════════════════════════════════════════════
#   Comparación — las seis configuraciones
# ═══════════════════════════════════════════════════════════════════════════

def comparacion(args):
    series = series_pool(args.results)
    out = args.out

    print(f"\n{'='*70}\n  COMPARACIÓN DE LAS {len(series)} CONFIGURACIONES\n"
          f"  Datos: {args.results}\n{'='*70}")

    # IGD+ y ε+ contra el frente de referencia común, el de las seis juntas.
    frente = load_reference_front(args.results)
    _generate_report(series, out, 'Comparación de las configuraciones', frente,
                     completo=args.completo)

    # Sin contraste estadístico: con 5 corridas por configuración no alcanza para hacer tests.
    print(f"\n✅ Comparación: {out}\n")



# ═══════════════════════════════════════════════════════════════════════════
#   Candidatos
#
#   Acá se juntan las seis configuraciones y se mira qué sobrevive y de dónde
#   sale.  No ordena algoritmos, caracteriza el pool.
# ═══════════════════════════════════════════════════════════════════════════

MOLECULAS_CSV = 'top15_docking.csv'

COLUMNAS_MOLECULAS = 5      # por fila de la figura



def _nota_pool(series):
    """Frase para el caption de la tabla del pool: con qué operadores entró cada
    algoritmo, leída de los metrics.csv de las series."""
    m = pd.concat([load_metrics(s.pop_dir) for s in series], ignore_index=True)
    cx = sorted(set(m['cx_prob']))
    mut = _mutacion_por_cruce(series)
    if len(cx) != 1 or not mut:
        return ''
    if len(mut) == 1:
        (cruce, valor), = mut.items()
        return (f'  Los algoritmos entran con cruce {cruce.upper()} (probabilidad '
                f'{_num_es(cx[0], "g")}) y mutación polinomial de probabilidad '
                f'{_num_es(valor, "g")} por gen.')
    return (f'  Cada algoritmo entra con dos ramas, una por familia de cruce, con '
            f'probabilidad de cruce {_num_es(cx[0], "g")} y mutación polinomial '
            f'con probabilidad por gen de {_texto_mutacion(series)}.')



def moleculas(args, out_dir):
    path = args.csv or os.path.join(args.results, MOLECULAS_CSV)
    if not os.path.exists(path):
        print(f"No existe {path}")
        return
    sel = pd.read_csv(path).sort_values('rank').reset_index(drop=True)

    ncols = COLUMNAS_MOLECULAS
    nrows = int(np.ceil(len(sel) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    for k, ax in enumerate(axes.flatten()):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#cccccc')
        if k >= len(sel):
            ax.set_visible(False)
            continue
        m = sel.iloc[k]
        img = render(m['smiles'])
        if img is not None:
            ax.imshow(img)
        ax.set_title(f"#{int(m['rank'])}  Dock {m['dock']:.2f}", fontsize=11,
                     fontweight='bold', pad=4)
        ax.set_xlabel(f"QED {m['qed']:.3f}  ·  SA {m['sa']:.2f}  "
                      f"·  Fsp3 {m['fsp3']:.2f}", fontsize=9, labelpad=3)

    fig.suptitle(f'Las {len(sel)} moléculas de mejor docking '
                 f'(todas cumplen Fsp3 ≥ {FSP3_MIN:g})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.5)
    png = os.path.join(out_dir, '6_moleculas_representativas.pdf')
    plt.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ 6_moleculas_representativas.pdf")

    sel.to_csv(os.path.join(out_dir, '6_moleculas_representativas.csv'), index=False)
    print("  ✓ 6_moleculas_representativas.csv")



FRENTES_COLS = ['smiles', 'qed', 'sa', 'fsp3', 'dock']



def _cols_frente(df):
    """FRENTES_COLS, más pChEMBL si el experimento lo optimizó."""
    return FRENTES_COLS + (['pchembl'] if 'pchembl' in df.columns else [])



def _n_corridas(series):
    """En cuántas corridas de las series apareció cada molécula."""
    partes = []
    for s in series:
        df = load_pareto_molecules(s.pop_dir)
        df['corrida'] = s.label + df['run']
        partes.append(df[['smiles', 'corrida']])
    return pd.concat(partes).groupby('smiles')['corrida'].nunique()



def frentes(series, pf_df, out_dir):
    """Los frentes en CSV: uno por configuración (frente_<ALG>_<cruce>.csv, sobre las
    veinte corridas y con SMILES únicos) y el general de todas (frente_general.csv),
    que además marca qué configuraciones hallaron cada molécula.  n_runs cuenta las
    corridas donde apareció."""
    os.makedirs(out_dir, exist_ok=True)
    por_serie, _, _ = _combined_pareto_fronts(series)
    for s in series:
        if s.label not in por_serie:
            continue
        f = por_serie[s.label][_cols_frente(por_serie[s.label])].copy()
        f['n_runs'] = f['smiles'].map(_n_corridas([s]))
        nombre = f'frente_{s.color_key}_{_cruce(s.label).lower()}.csv'
        f.sort_values('dock').to_csv(os.path.join(out_dir, nombre), index=False)

    at = atribuir_frente(series, pf_df, _por_serie)
    at['n_runs'] = at['smiles'].map(_n_corridas(series))
    cols = (_cols_frente(at) + ['n_runs'] + [c for c in at.columns if c.startswith('en_')]
            + ['origen'])
    at.sort_values('dock')[cols].to_csv(
        os.path.join(out_dir, 'frente_general.csv'), index=False)
    print(f"  ✓ frentes/  ({len(por_serie)} configuraciones y el general, "
          f"{len(at)} moléculas)")



def candidatos(args):
    series = series_pool(args.results)
    out = args.out
    tablas = os.path.join(out, 'tablas')
    figuras = os.path.join(out, 'figuras')
    os.makedirs(tablas, exist_ok=True)
    os.makedirs(figuras, exist_ok=True)

    print(f"\n{'='*70}\n  CANDIDATOS\n  Series: "
          f"{', '.join(s.label for s in series)}\n{'='*70}\n")

    pf_F, pf_df = build_reference_front(series)
    if pf_df is None:
        print("  ⚠ no se pudo construir el frente conjunto")
        return

    frentes(series, pf_df, os.path.join(out, 'frentes'))

    write_contribucion_table(series, pf_df, tablas, grupo_de=_por_serie,
                             etiqueta='Configuración', nota=_nota_pool(series))
    # Se agrupa por algoritmo: lo que interesa es quién puso cada molécula, no de
    # qué operador salió cada región.
    plot_frente_conjunto(series, figuras, '5_frente_conjunto.pdf', pf_df,
                         grupo_de=_algoritmo)

    moleculas(args, figuras)
    print(f"\n✅ Candidatos: {out}\n")
