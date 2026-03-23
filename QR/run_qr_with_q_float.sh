#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bin="./magma"
if [[ ! -x "$bin" ]]; then
  echo "Error: $bin not found or not executable. Build it first (e.g., make)." >&2
  exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"
log="qr_with_q_float_${ts}.log"
echo "Logging to: ${log}"
exec > >(tee -a "$log") 2>&1

sizes=(16384 20480 24576 28672 32768 36864)
warmup=2
repeat=5

csv="qr_with_q_float_${ts}.csv"
echo "CSV: ${csv}"
echo "n,geqrf_ms,geqrf_tflops,orgqr_ms,orgqr_tflops,total_ms,total_tflops" > "$csv"

run_one() {
  local n="$1"
  echo "==== float_${n} @ $(date +%F_%T) ===="
  local out
  out="$("$bin" "$n" float --with-q --warmup "$warmup" --repeat "$repeat")"
  echo "$out"
  local geqrf_ms geqrf_tflops orgqr_ms orgqr_tflops total_ms total_tflops
  geqrf_ms=$(printf "%s\n" "$out" | rg -o "MAGMA GEQRF takes ([0-9.]+) ms" -r '$1' | tail -n 1)
  geqrf_tflops=$(printf "%s\n" "$out" | rg -o "MAGMA GEQRF takes [0-9.]+ ms: ([0-9.]+) TFLOPs" -r '$1' | tail -n 1)
  orgqr_ms=$(printf "%s\n" "$out" | rg -o "MAGMA ORGQR takes ([0-9.]+) ms" -r '$1' | tail -n 1)
  orgqr_tflops=$(printf "%s\n" "$out" | rg -o "MAGMA ORGQR takes [0-9.]+ ms: ([0-9.]+) TFLOPs" -r '$1' | tail -n 1)
  total_ms=$(printf "%s\n" "$out" | rg -o "MAGMA GEQRF\\+ORGQR total ([0-9.]+) ms" -r '$1' | tail -n 1)
  total_tflops=$(printf "%s\n" "$out" | rg -o "MAGMA GEQRF\\+ORGQR total [0-9.]+ ms: ([0-9.]+) TFLOPs" -r '$1' | tail -n 1)
  echo "${n},${geqrf_ms:-},${geqrf_tflops:-},${orgqr_ms:-},${orgqr_tflops:-},${total_ms:-},${total_tflops:-}" >> "$csv"
  echo
}

for n in "${sizes[@]}"; do
  run_one "$n"
  sleep 1
done
