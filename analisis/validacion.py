"""
Los tres gráficos de validación del tutor (carpeta ANALISIS_MOLECULAS), reproducidos desde sus datos
sin recalcular nada pesado:

  admet_ranks.pdf   distribuciones KDE de 6 propiedades ADMET     <- df_combinado_admet.csv
  pchembl.pdf       distribución del pChEMBL predicho             <- fronts/*.csv + Random Forest
  posebuster.pdf    % de poses válidas en PoseBusters (top 100)   <- posebusters_top_pose0.csv

Salen en plots/exp5/validacion/.  Los colores están en PALETA, y el orden de las variantes en ORDEN.
Variantes: las seis configuraciones del experimento exp5_vae_pchembl_logp y los cuatro modelos
generativos (VAE, VAE con fine-tuning, GAN y GAN con fine-tuning).

Uso (entorno con seaborn y scikit-learn, p. ej. el de unidock):
    ~/miniforge3/envs/unidock/bin/python -m analisis.validacion
"""

import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ENTRADA = os.path.join(ROOT, 'ANALISIS_MOLECULAS')
SALIDA = os.path.join(ROOT, 'plots', 'exp5', 'validacion')

# ─── Colores: una entrada por variante ───────────────────────────────────────
PALETA = {
    'AGEMOEA_pcx_vae_pch_logp': '#2166AC',
    'AGEMOEA_sbx_vae_pch_logp': '#67A9CF',
    'NSGA2_pcx_vae_pch_logp':   '#1B7837',
    'NSGA2_sbx_vae_pch_logp':   '#7FBF7B',
    'NSGA3_pcx_vae_pch_logp':   '#B2182B',
    'NSGA3_sbx_vae_pch_logp':   '#EF8A62',
    'vae':                      '#762A83',
    'vae_ft':                   '#AF8DC3',
    'gan':                      '#D95F02',
    'gan_ft':                   '#FDB863',
}

# Estilo como en las figuras de analisis/: el color es el algoritmo y el cruce se distingue por el
# trazo (PCX sólido, SBX rayado).  Los cuatro modelos generativos llevan colores propios.
COLOR_ALGORITMO = {'NSGA2': '#E69F00', 'NSGA3': '#7B1FA2', 'AGEMOEA': '#008000'}   # = comun.COLORS
COLOR_MODELO = {'vae': '#D62728', 'vae_ft': '#F29B9B', 'gan': '#1F77B4', 'gan_ft': '#9ECAE1'}


def estilo(variante):
    """(color, rayado) de una variante: 'NSGA3_sbx_vae_pch_logp' → (violeta, '//')."""
    if variante in COLOR_MODELO:
        return COLOR_MODELO[variante], None
    algoritmo, cruce = variante.split('_')[:2]
    return COLOR_ALGORITMO[algoritmo], ('//' if cruce == 'sbx' else None)


# Orden de arriba hacia abajo en las distribuciones (ADMET).
ORDEN = ['AGEMOEA_sbx_vae_pch_logp', 'NSGA2_sbx_vae_pch_logp', 'NSGA3_sbx_vae_pch_logp',
         'AGEMOEA_pcx_vae_pch_logp', 'NSGA2_pcx_vae_pch_logp', 'NSGA3_pcx_vae_pch_logp',
         'vae', 'vae_ft', 'gan', 'gan_ft']

# Propiedades de la figura.  HIA_Hou y PAMPA_NCATS no están: casi todas las moléculas valen ~1 y las curvas
# no se distinguen, así que solo se informan en la tabla (ver PROPIEDADES_TABLA).
METRICAS_ADMET = ['Caco2_Wang', 'Bioavailability_Ma', 'Solubility_AqSolDB', 'ClinTox', 'LD50_Zhu', 'logP']


def admet():
    """Cuadrícula de 2 × 4 con una distribución KDE (ridgeline) por variante y propiedad."""
    df = pd.read_csv(os.path.join(ENTRADA, 'df_combinado_admet.csv'))
    for m in METRICAS_ADMET:
        df[m] = pd.to_numeric(df[m], errors='coerce')

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9), constrained_layout=True)
    for ax, metrica in zip(axes.flatten(), METRICAS_ADMET):
        for i, alg in enumerate(ORDEN):
            valores = df.loc[df['algoritmo'] == alg, metrica].dropna().to_numpy()
            if len(valores) < 2 or np.std(valores) == 0:      # la KDE pide dos valores distintos
                continue
            x = np.linspace(valores.min(), valores.max(), 300)
            dens = gaussian_kde(valores)(x)
            dens = dens / dens.max() * 0.8
            color, rayado = estilo(alg)
            ax.fill_between(x, i, i + dens, color=color, alpha=0.45, linewidth=0)
            if rayado:                  # SBX: el mismo color, rayado y línea discontinua
                ax.fill_between(x, i, i + dens, facecolor='none', edgecolor=color, hatch=rayado,
                                alpha=0.6, linewidth=0)
            ax.plot(x, i + dens, color=color, linewidth=1.5, linestyle='--' if rayado else '-')
        ax.set_title(metrica, fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(ORDEN)))
        ax.set_yticklabels([a.replace('_vae_pch_logp', '') for a in ORDEN], fontsize=14)
        ax.set_xlabel('Valor')
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle('Distribuciones KDE de las propiedades ADMET por algoritmo',
                 fontsize=16, fontweight='bold')
    # Leyenda en la esquina inferior derecha de la figura; constrained_layout le reserva el lugar.
    fig.legend(handles=[Patch(facecolor='#999999', edgecolor='white', label='PCX'),
                        Patch(facecolor='#999999', edgecolor='white', hatch='//', label='SBX')],
               loc='outside lower right', ncol=2, frameon=True)
    plt.savefig(os.path.join(SALIDA, 'admet_ranks.pdf'))
    plt.close(fig)
    print('  ✓ admet_ranks.pdf')


def _predicciones_pchembl():
    """Moléculas de cada variante con su pChEMBL predicho (columna predicted_pchembl)."""
    from pchembl import PChEMBLPredictor        # el mismo Random Forest del cuaderno del tutor
    predictor = PChEMBLPredictor()

    partes = []
    for alg in PALETA:
        df = pd.read_csv(os.path.join(ENTRADA, 'fronts', f'frente_{alg}.csv'))
        df['algoritmo'] = alg
        df['predicted_pchembl'] = predictor(df['smiles'].tolist())
        partes.append(df)
    df = pd.concat(partes, ignore_index=True)
    return df[df['predicted_pchembl'].notna() & (df['predicted_pchembl'] > 0)]


def pchembl():
    """Distribución del pChEMBL predicho por variante, con el umbral de actividad en 5."""
    df = _predicciones_pchembl()

    sns.set_theme(style='white', context='notebook')
    algoritmos = [a for a in PALETA if a in df['algoritmo'].unique()]
    fig, ax = plt.subplots(figsize=(10, 7))
    x_min, x_max = df['predicted_pchembl'].min(), df['predicted_pchembl'].max()
    x_grid = np.linspace(x_min, x_max, 500)

    for i, alg in enumerate(algoritmos):
        valores = df.loc[df['algoritmo'] == alg, 'predicted_pchembl'].dropna()
        if len(valores) < 2 or valores.nunique() < 2:
            continue
        dens = gaussian_kde(valores, bw_method='scott')(x_grid)
        dens = dens / dens.max() * 0.75
        color, rayado = estilo(alg)
        ax.fill_between(x_grid, i, i + dens, color=color, alpha=0.30, linewidth=0)
        if rayado:                      # SBX: el mismo color, rayado y línea discontinua
            ax.fill_between(x_grid, i, i + dens, facecolor='none', edgecolor=color, hatch=rayado,
                            alpha=0.55, linewidth=0)
        ax.plot(x_grid, i + dens, color=color, linewidth=1.5, linestyle='--' if rayado else '-')
        nombre = alg.replace('_vae_pch_logp', '')
        ax.text(x_max + 0.08, i + 0.15, f'{nombre} ({valores.mean():.2f})', va='center', fontsize=12)

    ax.axvline(x=5, color='black', linestyle='--', linewidth=1.5)
    ax.set_yticks([])
    ax.set_xlabel(r'$pChEMBL$ predicho')
    ax.set_ylabel('')
    ax.set_title('Distribución de pChEMBL predicho por algoritmo')
    leyenda = [Line2D([0], [0], color='black', linestyle='--', linewidth=1.5,
                      label=r'Umbral de actividad ($pChEMBL \geq 5$)'),
               Patch(facecolor='#999999', edgecolor='white', label='PCX'),
               Patch(facecolor='#999999', edgecolor='white', hatch='//', label='SBX')]
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.33)      # el margen de la derecha es para las etiquetas
    ax.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.17)      # espacio abajo para que la leyenda no pise el título del eje
    fig.legend(handles=leyenda, loc='lower right', ncol=3, frameon=True)   # esquina, sin quitar espacio
    plt.savefig(os.path.join(SALIDA, 'pchembl.pdf'), format='pdf')
    plt.close(fig)
    print('  ✓ pchembl.pdf')


def posebusters():
    """Porcentaje de poses que pasan todas las comprobaciones de PoseBusters, por variante."""
    df = pd.read_csv(os.path.join(ENTRADA, 'posebusters_top_pose0.csv'))
    comprobaciones = [c for c in df.columns
                      if c not in ('archivo', 'algorithm', 'pose_index', 'PBValidity')]
    df['PBValidity'] = df[comprobaciones].eq(True).all(axis=1)       # válida = pasa todas
    resumen = df.groupby('algorithm')['PBValidity'].agg(['count', 'mean']).reset_index()
    resumen['PBValidity_pct'] = resumen['mean'] * 100

    sns.set_theme(style='whitegrid', context='notebook')
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, fila in resumen.iterrows():
        color, rayado = estilo(fila['algorithm'])
        ax.bar(i, fila['PBValidity_pct'], color=color, hatch=rayado, edgecolor='white', linewidth=0)
    ax.set_xticks(range(len(resumen)))
    ax.set_xticklabels(resumen['algorithm'].str.replace('_vae_pch_logp', '', regex=False))
    ax.set_xlabel('Método')
    ax.set_ylabel('Poses válidas (%)')
    ax.set_title('PoseBusters validity en el top 100 de cada método')
    ax.set_ylim(0, 110)
    for contenedor in ax.containers:
        ax.bar_label(contenedor, fmt='%.1f%%', padding=3)
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    # Leyenda en la esquina inferior derecha de la figura, en horizontal: no quita espacio al gráfico.
    fig.legend(handles=[Patch(facecolor='#999999', edgecolor='white', label='PCX'),
                        Patch(facecolor='#999999', edgecolor='white', hatch='//', label='SBX')],
               loc='lower right', ncol=2, frameon=True)
    plt.savefig(os.path.join(SALIDA, 'posebuster.pdf'), format='pdf')
    plt.close(fig)
    print('  ✓ posebuster.pdf')


# ─── Tablas LaTeX de las figuras de ADMET y pChEMBL ───────────────────────────

# Propiedades de la tabla de ADMET: (columna, encabezado).  Se parten en dos tablas.
PROPIEDADES_TABLA = [('qed', 'QED'), ('sa', 'SA'), ('fsp3', 'Fsp3'), ('HIA_Hou', r'HIA\_Hou'),
                     ('Caco2_Wang', r'Caco2\_Wang'), ('PAMPA_NCATS', r'PAMPA\_NCATS'),
                     ('Bioavailability_Ma', r'Bioavailability\_Ma'),
                     ('Solubility_AqSolDB', r'Solubility\_AqSolDB'), ('ClinTox', 'ClinTox'),
                     ('LD50_Zhu', r'LD50\_Zhu'), ('logP', 'logP')]


def _escribir_tabla(df, columnas, encabezados, formato, caption, label, archivo):
    """Tabla LaTeX con una fila por variante (las seis configuraciones y, separados, los cuatro
    modelos generativos).  df: una fila por variante, indexada por su nombre."""
    lineas = ['\\begin{table}[htbp]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}',
              '\\begin{tabular}{l' + 'c' * len(columnas) + '}', '\\toprule',
              'Variante & ' + ' & '.join(encabezados) + ' \\\\', '\\midrule']
    for i, (variante, fila) in enumerate(df.iterrows()):
        if i == 6:                                       # después de las seis configuraciones
            lineas.append('\\midrule')
        nombre = variante.replace('_vae_pch_logp', '').replace('_', '\\_')
        lineas.append(nombre + ' & ' + ' & '.join(formato(fila[c], c) for c in columnas) + ' \\\\')
    lineas += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    with open(os.path.join(SALIDA, archivo), 'w') as f:
        f.write('\n'.join(lineas) + '\n')
    print(f'  ✓ {archivo}')


def tabla_admet():
    """Media de cada propiedad por variante, en dos tablas (la figura de ADMET las resume en curvas)."""
    df = pd.read_csv(os.path.join(ENTRADA, 'df_combinado_admet.csv'))
    columnas = [c for c, _ in PROPIEDADES_TABLA]
    for c in columnas:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    medias = df.groupby('algoritmo')[columnas].mean().reindex([a for a in PALETA if a in set(df['algoritmo'])])
    for parte, (desde, hasta) in enumerate([(0, 6), (6, len(columnas))], 1):
        cols = columnas[desde:hasta]
        _escribir_tabla(
            medias, cols, [h for c, h in PROPIEDADES_TABLA if c in cols], lambda x, c: f'${x:.3f}$',
            f'Propiedades ADMET de cada variante, parte {"I" if parte == 1 else "II"}.  Cada valor es la '
            f'media de la propiedad sobre las moléculas de la variante',
            f'tab:admet_resumen_{parte}', f'admet_ranks_{parte}.tex')


def tabla_pchembl():
    """Estadísticos descriptivos del pChEMBL predicho por variante."""
    df = _predicciones_pchembl()
    stats = (df.groupby('algoritmo')['predicted_pchembl'].describe()
             .reindex([a for a in PALETA if a in set(df['algoritmo'])]))
    columnas = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    encabezados = ['Moléculas', 'Media', 'Desv.', 'Mín.', 'Q1', 'Mediana', 'Q3', 'Máx.']
    _escribir_tabla(
        stats, columnas, encabezados, lambda x, c: f'${x:.0f}$' if c == 'count' else f'${x:.3f}$',
        'Estadísticos descriptivos del pChEMBL predicho de cada variante, sobre sus moléculas',
        'tab:pchembl_predicho', 'pchembl.tex')


# ─── Tabla contra los modelos de referencia ──────────────────────────────────

RESULTADOS = os.path.join(ROOT, 'results', 'exp5', 'exp5_vae_pchembl_logp')
BENCHMARKS = os.path.join(ROOT, 'benchmarks')
DOCK_WORST = 1.0            # puntaje de las moléculas cuyo docking falló: no entran al promedio
N_GENERADAS = 100_000       # moléculas que generó cada modelo de referencia
# (carpeta, nombre en la tabla, prefijo de sus archivos)
REFERENCIAS = [('vae_moses', 'VAE', 'vae'), ('vae_mrp1', 'VAE con fine-tuning', 'vae-ft'),
               ('gan_moses', 'GAN', 'gan'), ('gan_mrp1', 'GAN con fine-tuning', 'gan-ft')]
# Columnas: (clave, encabezado, decimales)
COLUMNAS_BENCH = [('qed', r'QED $\uparrow$', 3), ('sa', r'SA $\downarrow$', 3),
                  ('dock', r'Dock $\downarrow$', 3), ('pchembl', r'pChEMBL $\uparrow$', 3),
                  ('validez', r'Validez $\uparrow$', 3), ('unicidad', r'Unicidad $\uparrow$', 3),
                  ('novedad', r'Novedad $\uparrow$', 3)]


def _corridas(alg, cruce):
    """Una fila por corrida de la configuración: medias de su frente final y validez, unicidad
    (de la última generación) y novedad de la corrida."""
    filas = []
    patron = os.path.join(RESULTADOS, alg, f'{cruce}_pm', '*', 'run_*')
    for carpeta in sorted(glob.glob(patron)):
        frente = pd.read_csv(os.path.join(carpeta, 'molecules.csv'))
        met = pd.read_csv(os.path.join(carpeta, 'metrics.csv')).iloc[0]
        conv = pd.read_csv(os.path.join(carpeta, 'convergence.csv'))
        filas.append({'qed': frente.qed.mean(), 'sa': frente.sa.mean(),
                      'dock': frente[frente.dock < DOCK_WORST].dock.mean(),
                      'pchembl': frente.pchembl.mean(),
                      'validez': met.validity, 'unicidad': conv.uniqueness.iloc[-1],
                      'novedad': met.novelty})
    return pd.DataFrame(filas)


def _referencia(carpeta, prefijo, predictor, entrenamiento):
    """Medias y desviaciones (entre moléculas) de un modelo de referencia, y sus tres tasas de
    generación sobre las 100 000 moléculas que produjo."""
    df = pd.read_csv(os.path.join(BENCHMARKS, carpeta, f'{prefijo}_10k.csv'))
    df = df[df.valida.astype(bool)]
    df = df[df.dock < DOCK_WORST].assign(pchembl=lambda d: predictor(d.smiles.tolist()))[
        ['qed', 'sa', 'dock', 'pchembl']]
    with open(os.path.join(BENCHMARKS, carpeta, 'moleculas.smi')) as f:
        smi = [l.strip() for l in f if l.strip()]
    tasas = {'validez': len(smi) / N_GENERADAS, 'unicidad': len(set(smi)) / len(smi),
             'novedad': sum(s not in entrenamiento for s in smi) / len(smi)}
    return df.mean().to_dict() | tasas, df.std().to_dict()


def tabla_benchmarks():
    """Las seis configuraciones de exp5_vae_pchembl_logp contra los cuatro modelos de referencia.
    Configuraciones: media ± desviación entre sus corridas de la media del frente final.
    Referencias: media ± desviación entre moléculas; validez, unicidad y novedad sin desviación."""
    from pchembl import PChEMBLPredictor
    predictor = PChEMBLPredictor()
    entrenamiento = set(pd.read_pickle(os.path.join(ROOT, 'data', 'moses_train_smiles.pkl.gz'),
                                       compression='gzip'))
    nombres = {'NSGA2': 'NSGA-II', 'NSGA3': 'NSGA-III', 'AGEMOEA': 'AGE-MOEA'}

    filas = []
    for alg in ('NSGA2', 'NSGA3', 'AGEMOEA'):
        for cruce in ('pcx', 'sbx'):
            c = _corridas(alg, cruce)
            filas.append((f'{nombres[alg]} ({cruce.upper()})', c.mean().to_dict(), c.std().to_dict()))
    for carpeta, nombre, prefijo in REFERENCIAS:
        media, sd = _referencia(carpeta, prefijo, predictor, entrenamiento)
        filas.append((nombre, media, sd))

    # El mejor valor de cada columna (entre las diez filas) va en negrita; si varias filas empatan al
    # mostrar los decimales de la tabla, todas.
    mejor_de = {'qed': max, 'sa': min, 'dock': min, 'pchembl': max,
                'validez': max, 'unicidad': max, 'novedad': max}
    mejores = {k: mejor_de[k](round(m[k], d) for _, m, _ in filas) for k, _, d in COLUMNAS_BENCH}

    def celda(clave, dec, media, sd):
        texto = (f'{media[clave]:.{dec}f} \\pm {sd[clave]:.{dec}f}' if clave in sd
                 else f'{media[clave]:.{dec}f}')
        return f'$\\mathbf{{{texto}}}$' if round(media[clave], dec) == mejores[clave] else f'${texto}$'

    lineas = ['\\begin{table}[htbp]', '\\centering', '\\begin{tabular}{l' + 'c' * len(COLUMNAS_BENCH) + '}',
              '\\toprule', 'Modelo & ' + ' & '.join(h for _, h, _ in COLUMNAS_BENCH) + ' \\\\', '\\midrule']
    for i, (nombre, media, sd) in enumerate(filas):
        if i == 6:                                       # después de las seis configuraciones
            lineas.append('\\midrule')
        lineas.append(nombre + ' & ' + ' & '.join(celda(k, d, media, sd) for k, _, d in COLUMNAS_BENCH)
                      + ' \\\\')
    lineas += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    with open(os.path.join(SALIDA, 'comparacion_benchmarks.tex'), 'w') as f:
        f.write('\n'.join(lineas) + '\n')
    print('  ✓ comparacion_benchmarks.tex')


if __name__ == '__main__':
    os.makedirs(SALIDA, exist_ok=True)
    print(f'Gráficos y tablas de validación → {SALIDA}')
    admet()
    pchembl()
    posebusters()
    tabla_admet()
    tabla_pchembl()
    tabla_benchmarks()
