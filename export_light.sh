#!/usr/bin/env bash
# export_light.sh — copia ligera de results/ y results_baselines/ para transferir al PC.
# Excluye siempre all_molecules.csv.gz (log pesado) y, solo del grid, convergence.csv.
# Se corre EN el cluster.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# El log crudo se excluye siempre: es lo que pesa.  convergence.csv se excluye
# SOLO del grid, que son 513 configs x 20 semillas; las baselines son 80 corridas
# y lo necesitan, porque analisis.py detecta una serie por ese archivo
# (plot_comparison._has_runs) y sin el la etapa 4 no ve nada y no avisa por que.
EXCLUDES_GRID=(--exclude='all_molecules.csv.gz' --exclude='convergence.csv')
EXCLUDES_BASE=(--exclude='all_molecules.csv.gz')

for dir in results results_baselines; do
  src="$ROOT/$dir"
  dst="$ROOT/${dir}_light"
  if [[ "$dir" == results ]]; then
    EXCLUDES=("${EXCLUDES_GRID[@]}")
  else
    EXCLUDES=("${EXCLUDES_BASE[@]}")
  fi
  if [[ -d "$src" ]]; then
    echo "==> $dir  ->  ${dir}_light"
    rsync -a --prune-empty-dirs "${EXCLUDES[@]}" "$src/" "$dst/"
  else
    echo "(saltando $dir: no existe)"
  fi
done

echo
echo "Tamaños resultantes:"
du -sh "$ROOT/results_light" "$ROOT/results_baselines_light" 2>/dev/null || true
echo "Transfiere las carpetas: results_light/  y  results_baselines_light/"
