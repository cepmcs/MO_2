#!/usr/bin/env bash
# Copia las 17 configuraciones ganadoras de la etapa 1 a una carpeta aparte,
# conservando la estructura de directorios original.
#
# Uso:
#   ./copiar_ganadoras.sh                      # results/ -> winners/
#   ./copiar_ganadoras.sh results_light        # otra carpeta origen
#   ./copiar_ganadoras.sh results winners      # origen y destino explícitos
#   DRY_RUN=1 ./copiar_ganadoras.sh            # solo listar, no copiar

set -euo pipefail

SRC="${1:-results}"
DST="${2:-winners}"
DRY_RUN="${DRY_RUN:-0}"

# Las 17 ganadoras (de plots_hp/selected_configs.csv).
# GA:    <ALG>/<cruce>_<mutacion>/cx<prob>_mut<prob>_pop<N>_gen<G>
# MOPSO: MOPSO/pop<N>_gen<G>_w<w>_c1<c1>_c2<c2>
WINNERS=(
  "NSGA2/pcx_pm/cx1_mut0.004_pop400_gen250"
  "NSGA2/pcx_gauss/cx1_mut0.012_pop400_gen250"
  "NSGA2/sbx_pm/cx1_mut0.031_pop400_gen250"
  "NSGA2/sbx_gauss/cx1_mut0.031_pop400_gen250"

  "NSGA3/pcx_pm/cx1_mut0.031_pop400_gen250"
  "NSGA3/pcx_gauss/cx1_mut0.012_pop400_gen250"
  "NSGA3/sbx_pm/cx1_mut0.012_pop400_gen250"
  "NSGA3/sbx_gauss/cx0.9_mut0.031_pop400_gen250"

  "MOEAD/pcx_pm/cx1_mut0.031_pop400_gen250"
  "MOEAD/pcx_gauss/cx1_mut0.031_pop400_gen250"
  "MOEAD/sbx_pm/cx1_mut0.031_pop400_gen250"
  "MOEAD/sbx_gauss/cx0.9_mut0.031_pop400_gen250"

  "AGEMOEA/pcx_pm/cx1_mut0.031_pop400_gen250"
  "AGEMOEA/pcx_gauss/cx1_mut0.031_pop400_gen250"
  "AGEMOEA/sbx_pm/cx1_mut0.031_pop200_gen500"
  "AGEMOEA/sbx_gauss/cx0.9_mut0.031_pop100_gen1000"

  "MOPSO/pop100_gen1000_w0.6_c11.5_c22"
)

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: no existe la carpeta origen '$SRC'" >&2
  exit 1
fi

echo "origen : $SRC"
echo "destino: $DST"
[[ "$DRY_RUN" == "1" ]] && echo "MODO DRY-RUN (no copia nada)"
echo

ok=0
faltan=()
total_runs=0

for w in "${WINNERS[@]}"; do
  src_dir="$SRC/$w"
  if [[ ! -d "$src_dir" ]]; then
    printf '  FALTA  %s\n' "$w"
    faltan+=("$w")
    continue
  fi

  n_runs=$(find "$src_dir" -maxdepth 1 -type d -name 'run_*' | wc -l)
  size=$(du -sh "$src_dir" 2>/dev/null | cut -f1)
  printf '  ok     %-52s %3d runs  %6s\n' "$w" "$n_runs" "$size"

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$DST/$(dirname "$w")"
    cp -a "$src_dir" "$DST/$(dirname "$w")/"
  fi

  ok=$((ok + 1))
  total_runs=$((total_runs + n_runs))
done

echo
echo "configuraciones copiadas: $ok/${#WINNERS[@]}   (corridas: $total_runs)"

if [[ ${#faltan[@]} -gt 0 ]]; then
  echo
  echo "NO SE ENCONTRARON ${#faltan[@]}:"
  printf '  %s\n' "${faltan[@]}"
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
  echo "tamaño total en $DST: $(du -sh "$DST" | cut -f1)"
fi
