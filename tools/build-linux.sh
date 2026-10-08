#!/bin/sh
# Build libaipudrv (Zhouyi v3, hardware) for Linux, on FreeBSD: Rocky Linux's
# g++ under the Linuxulator (linux-rl9-devtools), for Linux programs (CIX's
# libnoe) run under the Linuxulator.  Otherwise as build-freebsd.sh:
# build-linux/umd for objects, build-linux/bin for the libraries.  Needs
# gmake.  Usage: tools/build-linux.sh [make args]
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export COMPASS_DRV_BTENVAR_UMD_BUILD_DIR=$ROOT/build-linux/umd
export BUILD_AIPU_DRV_ODIR=$ROOT/build-linux/bin
export COMPASS_DRV_BTENVAR_UMD_V_MAJOR=6
export COMPASS_DRV_BTENVAR_UMD_V_MINOR=0.0
export COMPASS_DRV_BTENVAR_UMD_SO_NAME=libaipudrv.so
export COMPASS_DRV_BTENVAR_UMD_SO_NAME_MAJOR=libaipudrv.so.6
export COMPASS_DRV_BTENVAR_UMD_SO_NAME_FULL=libaipudrv.so.6.0.0
export COMPASS_DRV_BTENVAR_UMD_A_NAME=libaipudrv.a
export COMPASS_DRV_BTENVAR_UMD_A_NAME_FULL=libaipudrv.a.6.0.0
mkdir -p "$COMPASS_DRV_BTENVAR_UMD_BUILD_DIR" "$BUILD_AIPU_DRV_ODIR"
# -mno-outline-atomics: the port's libgcc_s.so is a symlink to libgcc_s.so.1,
# where Rocky's is a linker script that adds libgcc.a, so a shared library
# would be left needing libgcc.a's atomics helpers (__aarch64_ldadd4_acq_rel).
exec gmake -C "$ROOT/umd" standard_api BUILD_AIPU_VERSION=aipu_v3 \
    BUILD_TARGET_PLATFORM=hw \
    CXX="/compat/linux/usr/bin/g++ -mno-outline-atomics" "$@"
