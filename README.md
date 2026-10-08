# aipu-umd

Arm China's Zhouyi NPU user-mode driver (`libaipudrv`, the "standard API")
for FreeBSD, for the CIX Sky1's NPU (Zhouyi X2, architecture v3; Orange Pi
6 Plus, Radxa Orion O6), with the kernel driver from
[aipu-kmod](https://github.com/JamesKane/aipu-kmod).

- `umd/`, `samples/`: Arm China's user-mode driver and samples, release
  **4.1.0** (Apache-2.0;
  [Compass_NPU_Driver](https://github.com/Arm-China/Compass_NPU_Driver),
  commit f35bac0), imported unmodified, then changed for FreeBSD.
- `freebsd/include/`: the `linux/*.h` the kernel's ioctl header includes.
- `tools/build-freebsd.sh`: builds `libaipudrv` natively (clang, gmake).
- `tools/build-linux.sh`: builds it for Linux programs under the
  Linuxulator, with Rocky Linux's g++ (`linux-rl9-devtools`).
- `tools/npurun/`: runs a compiled graph on preprocessed images and ranks
  the output (BSD-2-Clause).
- `tools/noerun/`: the same through CIX's binary-only `libnoe`, a Linux
  program (BSD-2-Clause).

## Why 4.1.0

The library and the kernel driver share the ioctl structures
(`include/kmd/armchina_aipu.h`). CIX's kernel driver 6.2.0, which
aipu-kmod ports, is Arm China's 4.1.0 driver with a Sky1 layer and two
cache ioctls: its ioctl header, TCB layout and v3 definitions are 4.1.0's.
Arm China 4.2.0 and later changed `struct aipu_buf_desc` and
`struct aipu_job_desc`, so their library does not work with it. The
library here builds against the kernel driver's own header.

## Building and running (on the board)

    pkg install gmake
    sh tools/build-freebsd.sh            # build/bin/libaipudrv.so.6.0.0
    c++ -std=c++14 -O2 -Iumd/include -o npurun tools/npurun/npurun.cpp \
        -Lbuild/bin -laipudrv -Wl,-rpath,$PWD/build/bin
    ./npurun graph.elf labels.txt image.f32 ...

Compiled models: CIX's AI model hub on ModelScope
(`cix/ai_model_hub_26_Q2`, no login). A `.cix` file wraps the ELF graph
`libaipudrv` loads: the graph starts at the `\x7fELF` magic and its length
is the 32-bit little-endian word just before it. MobileNetV2's input is
1x3x224x224 (NCHW, RGB, resized, scaled to 0-1, ImageNet mean/std);
`npurun` quantizes it with the scale the graph reports.

On the Orange Pi 6 Plus under FreeBSD, CIX's MobileNetV2 labels ImageNet
validation images right in about 7.3 ms each.

## CIX's libnoe, under the Linuxulator

CIX's NOE API (`libnoe.so.3`, from its `cix-noe-umd` 3.1.2 package; glibc
2.34) is a layer over `libaipudrv`, which it loads with `dlopen`. Linux
programs reach `/dev/aipu` through `aipu_linux.ko` (in aipu-kmod), so
`libnoe` runs on a Linux `libaipudrv` built here:

    pkg install linux_base-rl9 linux-rl9-devtools gmake
    sysrc linux_enable=YES && service linux start
    kldload aipu_linux
    sh tools/build-linux.sh               # build-linux/bin
    /compat/linux/usr/bin/g++ -std=c++14 -O2 -I<cix include/npu> \
        -o noerun tools/noerun/noerun.cpp -L<libnoe dir> -lnoe
    AIPU_LIB_PATH=$PWD/build-linux/bin/libaipudrv.so \
        LD_LIBRARY_PATH=<libnoe dir> ./noerun model.cix labels.txt image.f32

`AIPU_LIB_PATH` names the library file (not its directory); without it,
`libnoe` searches the library path for `libaipudrv.so`. `libnoe` reads
`.cix` files itself. Two things to know:

- `noe_create_job()` needs a configuration: `libnoe` 3.1.2 dereferences it
  although the header defaults it to `nullptr`.
- The port's `libgcc_s.so` is a symlink, not Rocky's linker script (which
  adds `libgcc.a`), so `build-linux.sh` builds with `-mno-outline-atomics`.
