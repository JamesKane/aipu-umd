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
  program, and `noerun.py` through its Python wheel, from JPEGs
  (BSD-2-Clause).
- `tools/ort/`: ONNX models on the NPU through CIX's ONNX Runtime
  execution provider: `prepare.py` readies and quantizes a model,
  `ortrun.py` classifies JPEGs (BSD-2-Clause).

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

## CIX's Python wheel

`libnoe`'s wheel (`libnoe-3.1.2-py3-none-manylinux2014_aarch64.whl`, in the
same package) holds extension modules for CPython 3.10 to 3.13. Rocky 9's
Linux Python is 3.9, so the interpreter is a Linux CPython 3.12 from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
(`aarch64-unknown-linux-gnu`, unpacked in a directory of its own), with
numpy and Pillow from pip:

    python/bin/python3.12 -m pip install numpy pillow libnoe-3.1.2-*.whl
    AIPU_LIB_PATH=$PWD/build-linux/bin/libaipudrv.so \
        python/bin/python3.12 tools/noerun/noerun.py model.cix labels.txt *.JPEG

- Not from a directory that holds `libnoe.so` (the C library): Python
  imports it as the module.
- `noe_load_tensor()` takes the input as `bytes`: given an ndarray, the
  wheel returns outputs as ndarrays that repeat their first element.

## CIX's ONNX Runtime provider

`cix-npu-onnxruntime` 1.2.0 (in
[radxa-pkg/cix-prebuilt](https://github.com/radxa-pkg/cix-prebuilt)'s
release `26Q2-2607`, with `cix-noe-umd` 3.1.2) is ONNX Runtime 1.22 with a
`ZhouyiExecutionProvider`, whose compiler (Arm China's Compass, binary) turns
the model into an NPU graph on the board when a session starts. Everything
it needs comes in its wheels and libraries, including its own `libaipudrv`
build (`libaipu_driver.so`), which matches aipu-kmod's ioctls. Under the
Linuxulator it needs:

- a Linux CPython **3.11** (the wheel is `cp311`), python-build-standalone's;
- libstdc++ with `GLIBCXX_3.4.30`, newer than Rocky 9's: conda-forge's
  `libstdcxx` and `libgcc` 14 (built for glibc 2.17), in a directory of
  their own on `LD_LIBRARY_PATH` (Debian 12's need glibc 2.36);
- `OPERATOR_PATH` naming the package's `operator/` directory (the operator
  libraries the compiler links), else it looks in `./operator`.

```
python/bin/python3.11 -m pip install numpy pillow onnx \
    onnxruntime_zhouyi-1.22.0-cp311-cp311-linux_aarch64.whl \
    ZhouyiOperators_x2-25.9.19-py3-none-any.whl
export LD_LIBRARY_PATH=<libstdc++ dir> OPERATOR_PATH=<package>/operator
python/bin/python3.11 tools/ort/prepare.py mobilenetv2-7.onnx int8.onnx *.JPEG
python/bin/python3.11 tools/ort/ortrun.py int8.onnx labels.txt *.JPEG
```

The compiler takes **int8** models (QDQ): float depthwise convolutions, and
float convolutions on larger tensors, fail to compile. `prepare.py`
quantizes with ONNX Runtime's own tools, calibrated on JPEGs, after three
fixes the compiler needs:

- BatchNormalization folded into the convolutions (ONNX Runtime's
  quantization pre-processing): a lone quantized BatchNormalization comes
  out wrong, silently. For the folding, weights must not be graph inputs, as
  older exporters (opset 7) make them.
- Reshape shapes with 0 ("copy this dimension") made explicit: the compiler
  takes the 0 for a size.
- Opset 13, for per-channel quantization.

CIX's ONNX-model-zoo MobileNetV2 (`mobilenetv2-7.onnx`, the model its `.cix`
is built from) then labels CIX's five test images right on the NPU, ~13.8 ms
each, the session compiling in ~1.2 s; float inputs and outputs are
converted on the CPU, and the calibration is ONNX Runtime's, not CIX's.
