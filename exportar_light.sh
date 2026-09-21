#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Empaqueta de results/ solo lo liviano de cada corrida, para bajarlo al PC:
#    molecules.csv   el frente de la corrida (lo que usa el frente de referencia)
#    metrics.csv     sus métricas (HV, validez, ...), para las tablas
#  Deja afuera all_molecules.csv.gz, que es casi todo el peso.
#  Conserva el árbol <ALG>/<combo>/<config>/run_XX/.
#
#    bash exportar_light.sh              # usa ./results
#    bash exportar_light.sh otra/carpeta
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

RES=${1:-results}
TAR=results_light.tar.gz

[ -d "$RES" ] || { echo "ERROR: no existe $RES" >&2; exit 1; }

N=$(find "$RES" -path '*/run_*/molecules.csv' | wc -l)
[ "$N" -gt 0 ] || { echo "ERROR: no hay run_*/molecules.csv en $RES" >&2; exit 1; }

# Rutas relativas a $RES, así el tar se extrae donde uno quiera.
( cd "$RES" && find . -path '*/run_*/molecules.csv' -o -path '*/run_*/metrics.csv' ) \
    | sed 's|^\./||' | tar czf "$TAR" -C "$RES" -T -

echo "Corridas: $N"
for d in "$RES"/*/; do
    echo "  $(basename "$d"): $(find "$d" -name molecules.csv | wc -l)"
done
echo
echo "✅ $TAR  ($(du -h "$TAR" | cut -f1))"
echo
echo "En el PC, desde la raíz del repo:"
echo "   scp $(hostname):$(pwd)/$TAR ."
echo "   tar xzf $TAR -C results/main/grid"
