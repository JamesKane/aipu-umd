#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
#
# Classify JPEGs with an ImageNet ONNX model through ONNX Runtime, on CIX's
# Zhouyi execution provider (the NPU) or, with --cpu, on the CPU.  For an
# int8 model from prepare.py; the provider compiles it when the session
# starts.
#
# Usage: ortrun.py [--cpu] model.onnx labels.txt image.jpg ...

import re
import sys
import time

import numpy as np
from PIL import Image
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


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
    return x.transpose(2, 0, 1)[None]  # NCHW


def main():
    args = sys.argv[1:]
    cpu = args[:1] == ["--cpu"]
    if cpu:
        args = args[1:]
    if len(args) < 3:
        sys.exit("usage: ortrun.py [--cpu] model.onnx labels.txt image.jpg ...")
    labels = read_labels(args[1])
    providers = ["CPUExecutionProvider"]
    if not cpu:
        providers.insert(0, "ZhouyiExecutionProvider")
    t0 = time.perf_counter()
    s = ort.InferenceSession(args[0], providers=providers)
    print("session: %.1f s, %s" % (time.perf_counter() - t0,
                                   ", ".join(s.get_providers())))
    inp = s.get_inputs()[0]
    for path in args[2:]:
        x = preprocess(path, inp.shape[-1])
        t0 = time.perf_counter()
        y = s.run(None, {inp.name: x})[0].ravel()
        t1 = time.perf_counter()
        print(f"{path}: {(t1 - t0) * 1000:.2f} ms")
        for i in np.argsort(-y)[:5]:
            print(f"  {i:3d} {y[i]:7.3f} {labels.get(int(i), '')}")


if __name__ == "__main__":
    main()
