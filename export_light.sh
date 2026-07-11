#!/usr/bin/env bash
# export_light.sh — copia ligera de results/ y results_baselines/ para transferir al PC.
# Excluye all_molecules.csv.gz (log pesado) y convergence.csv. Se corre EN el cluster.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXCLUDES=(--exclude='all_molecules.csv.gz' --exclude='convergence.csv')

for dir in results results_baselines; do
  src="$ROOT/$dir"
  dst="$ROOT/${dir}_light"
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
