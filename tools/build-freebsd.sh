#!/bin/sh
# Build libaipudrv (Zhouyi v3, hardware) natively on FreeBSD, as Arm China's
# bash_env_setup.sh + build_all.sh would: build/umd for objects, build/bin
# for the libraries.  Needs gmake.  Usage: tools/build-freebsd.sh [make args]
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export COMPASS_DRV_BTENVAR_UMD_BUILD_DIR=$ROOT/build/umd
export BUILD_AIPU_DRV_ODIR=$ROOT/build/bin
export COMPASS_DRV_BTENVAR_UMD_V_MAJOR=6
export COMPASS_DRV_BTENVAR_UMD_V_MINOR=0.0
export COMPASS_DRV_BTENVAR_UMD_SO_NAME=libaipudrv.so
export COMPASS_DRV_BTENVAR_UMD_SO_NAME_MAJOR=libaipudrv.so.6
export COMPASS_DRV_BTENVAR_UMD_SO_NAME_FULL=libaipudrv.so.6.0.0
export COMPASS_DRV_BTENVAR_UMD_A_NAME=libaipudrv.a
export COMPASS_DRV_BTENVAR_UMD_A_NAME_FULL=libaipudrv.a.6.0.0
mkdir -p "$COMPASS_DRV_BTENVAR_UMD_BUILD_DIR" "$BUILD_AIPU_DRV_ODIR"
exec gmake -C "$ROOT/umd" standard_api BUILD_AIPU_VERSION=aipu_v3 \
    BUILD_TARGET_PLATFORM=hw CXX=c++ "$@"
