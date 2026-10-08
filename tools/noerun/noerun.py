#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
#
# noerun.cpp in Python, through CIX's libnoe wheel, from JPEGs: each image
# preprocessed as CIX's model hub does for ImageNet (RGB, resized to the
# input, scaled to 0-1, ImageNet mean/std, NCHW), quantized with the scale
# and zero point the graph reports, run, and the output dequantized and
# ranked.  For a Linux CPython 3.10-3.13 under the Linuxulator, with numpy and
# Pillow; AIPU_LIB_PATH names the Linux libaipudrv.so (tools/build-linux.sh).
#
# Usage: noerun.py model.cix labels.txt image.jpg ...

import re
import sys
import time

import numpy as np
from PIL import Image

import libnoe

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TYPES = {libnoe.NOE_DATA_TYPE_S8: np.int8, libnoe.NOE_DATA_TYPE_U8: np.uint8}


def check(npu, ret, what):
    status, *value = ret if isinstance(ret, tuple) else (ret,)
    if status != libnoe.NOE_STATUS_SUCCESS:
        sys.exit(f"{what}: {npu.noe_get_error_message(int(status))}")
    return value[0] if value else None


def read_labels(path):
    # Python dict lines: "N: 'name, other names',".
    labels = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"\{?\s*(\d+)\s*:\s*'(.*)'", line)
            if m:
                labels[int(m.group(1))] = m.group(2)
    return labels


def preprocess(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)  # HWC to CHW


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: noerun.py model.cix labels.txt image.jpg ...")
    labels = read_labels(sys.argv[2])
    npu = libnoe.NPU()
    check(npu, npu.noe_init_context(), "init_context")
    graph = check(npu, npu.noe_load_graph(sys.argv[1]), "load_graph")
    ind = npu.noe_get_tensor_descriptor(graph, libnoe.NOE_TENSOR_TYPE_INPUT, 0)
    outd = npu.noe_get_tensor_descriptor(graph, libnoe.NOE_TENSOR_TYPE_OUTPUT,
                                         0)
    if ind.data_type not in TYPES or outd.data_type not in TYPES:
        sys.exit("only 8-bit tensors handled")
    size = int(round((ind.size / 3) ** 0.5))  # square RGB, 8 bits
    job = check(npu, npu.noe_create_job(graph, libnoe.noe_create_job_cfg_t()),
                "create_job")

    for path in sys.argv[3:]:
        info = np.iinfo(TYPES[ind.data_type])
        q = np.clip(np.rint(preprocess(path, size) * ind.scale) +
                    ind.zero_point, info.min, info.max)
        # As bytes: given an ndarray, libnoe 3.1.2's wheel returns outputs
        # as ndarrays repeating their first element.
        check(npu, npu.noe_load_tensor(job, 0,
              q.astype(TYPES[ind.data_type]).tobytes()), "load_tensor")
        t0 = time.perf_counter()
        check(npu, npu.noe_job_infer_sync(job, 5000), "job_infer_sync")
        t1 = time.perf_counter()
        raw = check(npu, npu.noe_get_tensor(job, libnoe.NOE_TENSOR_TYPE_OUTPUT,
                    0), "get_tensor")
        y = (np.frombuffer(raw, TYPES[outd.data_type]).astype(np.float32) -
             outd.zero_point) / outd.scale
        print(f"{path}: {(t1 - t0) * 1000:.2f} ms")
        for i in np.argsort(-y)[:5]:
            print(f"  {i:3d} {y[i]:7.3f} {labels.get(int(i), '')}")

    check(npu, npu.noe_clean_job(job), "clean_job")
    check(npu, npu.noe_unload_graph(graph), "unload_graph")
    check(npu, npu.noe_deinit_context(), "deinit_context")


if __name__ == "__main__":
    main()
