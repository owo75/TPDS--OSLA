#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bin="./magma"
if [[ ! -x "$bin" ]]; then
  echo "Error: $bin not found or not executable. Build it first (e.g., make magma)." >&2
  exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"
log="magma_sweep_${ts}.log"
echo "Logging to: ${log}"
exec > >(tee -a "$log") 2>&1

# float_sizes=(16384 24576 32768 40960 49152 57344 65536)
double_sizes=(24576 32768 40960 49152 51200)

run_one() {
  local n="$1"
  local dtype="$2"
  local label="${dtype}_${n}"
  echo "==== ${label} @ $(date +%F_%T) ===="
  "$bin" "$n" "$dtype"
  echo
}

for n in "${float_sizes[@]}"; do
  run_one "$n" float
  sleep 1
done

for n in "${double_sizes[@]}"; do
  run_one "$n" double
  sleep 1
done
