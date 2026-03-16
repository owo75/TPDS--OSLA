#!/usr/bin/env bash
set -u

cd "$(dirname "$0")" || exit 1

SIZES=(16384 24576 32768 40960 46080)
LOG_FILE="magma_float_bench.log"
CSV_FILE="magma_float_bench.csv"
MAGMA_ROOT="${MAGMA_ROOT:-/home/sl/nfs_data/sl/magma-2.9.0/magma_install}"

echo "== MAGMA float benchmark ==" | tee "$LOG_FILE"
echo "time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "MAGMA_ROOT: $MAGMA_ROOT" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Always rebuild to avoid stale linkage to a different MAGMA installation.
echo "building magma benchmark..." | tee -a "$LOG_FILE"
MAGMA_DEFS=""
if [[ "$MAGMA_ROOT" == *"ilp64"* ]]; then
  MAGMA_DEFS="-DMKL_ILP64"
  SIZES+=(49152 57344 65536)
fi
echo "sizes: ${SIZES[*]}" | tee -a "$LOG_FILE"
echo "MAGMA_DEFS: ${MAGMA_DEFS:-<none>}" | tee -a "$LOG_FILE"
make magma -B MAGMA_ROOT="$MAGMA_ROOT" MAGMA_DEFS="$MAGMA_DEFS" 2>&1 | tee -a "$LOG_FILE"

if [[ "$MAGMA_ROOT" == *"ilp64"* ]]; then
  if [[ -n "${MKLROOT:-}" ]]; then
    export LD_LIBRARY_PATH="${MKLROOT}/lib/intel64:${MKLROOT}/lib:${LD_LIBRARY_PATH:-}"
  fi
fi

export LD_LIBRARY_PATH="${MAGMA_ROOT}/lib:${LD_LIBRARY_PATH:-}"

missing_libs="$(ldd ./magma 2>/dev/null | grep 'not found' || true)"
if [[ -n "$missing_libs" ]]; then
  echo "ERROR: unresolved shared libraries:" | tee -a "$LOG_FILE"
  echo "$missing_libs" | tee -a "$LOG_FILE"
  echo "Hint: for ILP64 build, source oneAPI first: source /opt/intel/oneapi/setvars.sh" | tee -a "$LOG_FILE"
  exit 1
fi

echo "n,status,ms,tflops" > "$CSV_FILE"

for n in "${SIZES[@]}"; do
  echo ">>> Running: ./magma $n float" | tee -a "$LOG_FILE"
  output="$(./magma "$n" float 2>&1)"
  rc=$?
  echo "$output" | tee -a "$LOG_FILE"

  if [[ $rc -ne 0 ]]; then
    echo "$n,FAIL,," >> "$CSV_FILE"
    echo "result: FAIL (exit code $rc)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    continue
  fi

  line="$(echo "$output" | grep 'MAGMA Cholesky takes' | tail -n 1)"
  ms="$(echo "$line" | awk '{print $4}')"
  tflops="$(echo "$line" | awk '{print $6}')"
  echo "$n,OK,$ms,$tflops" >> "$CSV_FILE"
  echo "result: OK, ms=$ms, tflops=$tflops" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
done

echo "Done."
echo "log: $LOG_FILE"
echo "csv: $CSV_FILE"
