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
- `tools/npurun/`: runs a compiled graph on preprocessed images and ranks
  the output (BSD-2-Clause).

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
