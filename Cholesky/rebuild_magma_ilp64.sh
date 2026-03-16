#!/usr/bin/env bash
set -euo pipefail

MAGMA_SRC="${MAGMA_SRC:-/home/sl/nfs_data/sl/magma-2.9.0}"
BUILD_DIR="${BUILD_DIR:-$MAGMA_SRC/build_ilp64}"
INSTALL_DIR="${INSTALL_DIR:-$MAGMA_SRC/magma_install_ilp64}"
CUDA_ARCHS="${CUDA_ARCHS:-80;89;90}"

if [[ -z "${MKLROOT:-}" ]]; then
  echo "ERROR: MKLROOT is not set."
  echo "Please load oneAPI MKL first, e.g.:"
  echo "  source /opt/intel/oneapi/setvars.sh"
  echo "Then rerun this script."
  exit 1
fi

MKL_LIB_DIR="${MKLROOT}/lib/intel64"
if [[ ! -d "$MKL_LIB_DIR" ]]; then
  MKL_LIB_DIR="${MKLROOT}/lib"
fi

MKL_ILP64_LIB="$MKL_LIB_DIR/libmkl_intel_ilp64.so"
MKL_SEQ_LIB="$MKL_LIB_DIR/libmkl_sequential.so"
MKL_CORE_LIB="$MKL_LIB_DIR/libmkl_core.so"

for f in "$MKL_ILP64_LIB" "$MKL_SEQ_LIB" "$MKL_CORE_LIB"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing MKL library: $f"
    exit 1
  fi
done

echo "Configuring MAGMA ILP64 build..."
echo "  src     : $MAGMA_SRC"
echo "  build   : $BUILD_DIR"
echo "  install : $INSTALL_DIR"
echo "  archs   : $CUDA_ARCHS"

cmake -S "$MAGMA_SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
  -DCMAKE_C_FLAGS="-DMKL_ILP64" \
  -DCMAKE_CXX_FLAGS="-DMKL_ILP64" \
  -DCMAKE_CUDA_FLAGS="-DMKL_ILP64" \
  -DBLA_VENDOR=Intel10_64ilp \
  -DBLAS_LIBRARIES="$MKL_ILP64_LIB;$MKL_SEQ_LIB;$MKL_CORE_LIB" \
  -DLAPACK_LIBRARIES="$MKL_ILP64_LIB;$MKL_SEQ_LIB;$MKL_CORE_LIB"

cmake --build "$BUILD_DIR" -j"$(nproc)"
cmake --install "$BUILD_DIR"

echo
echo "Done. ILP64 MAGMA installed at:"
echo "  $INSTALL_DIR"
echo
echo "Use it in your benchmark build:"
echo "  make magma -B MAGMA_ROOT=$INSTALL_DIR"
