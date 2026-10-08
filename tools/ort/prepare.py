#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
#
# Prepare a float ImageNet ONNX model for CIX's Zhouyi execution provider
# (ONNX Runtime 1.22, cix-npu-onnxruntime), which compiles it for the NPU when
# a session starts.  Its compiler takes int8 models: float depthwise and
# larger float convolutions fail to compile.  So:
#
#  1. opset 13 at least (per-channel quantization), and weights out of the
#     graph inputs (older exporters list them there), where ONNX Runtime
#     would take them for variables and fold nothing;
#  2. ONNX Runtime's quantization pre-processing, which folds
#     BatchNormalization into the convolutions: the compiler gets a lone
#     quantized BatchNormalization wrong;
#  3. Reshape shapes with 0 ("copy this dimension") made explicit: the
#     compiler reads the 0 as a size;
#  4. static int8 quantization (QDQ, per channel), calibrated on JPEGs
#     preprocessed as CIX's model hub does for ImageNet.
#
# Usage: prepare.py in.onnx out.onnx image.jpg ...

import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference, version_converter
from PIL import Image
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]  # NCHW


class Reader(CalibrationDataReader):
    def __init__(self, name, size, paths):
        self.it = iter([{name: preprocess(p, size)} for p in paths])

    def get_next(self):
        return next(self.it, None)


def explicit_reshapes(m):
    """Replace 0s in constant Reshape shapes by the input's dimension."""
    m = shape_inference.infer_shapes(m)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(m.graph.value_info) + list(m.graph.input)}
    inits = {t.name: t for t in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type != "Reshape" or n.input[1] not in inits:
            continue
        shape = numpy_helper.to_array(inits[n.input[1]]).copy()
        src = shapes.get(n.input[0])
        if 0 not in shape or src is None:
            continue
        for i in np.flatnonzero(shape == 0):
            shape[i] = src[i]
        inits[n.input[1]].CopyFrom(numpy_helper.from_array(shape,
                                                           n.input[1]))
        print("Reshape %s: shape %s" % (n.name or n.output[0], shape.tolist()))
    return m


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: prepare.py in.onnx out.onnx image.jpg ...")
    src, dst, images = sys.argv[1], sys.argv[2], sys.argv[3:]
    with tempfile.TemporaryDirectory() as tmp:
        m = onnx.load(src)
        if max(o.version for o in m.opset_import
               if o.domain in ("", "ai.onnx")) < 13:
            m = version_converter.convert_version(m, 13)
        m.ir_version = max(m.ir_version, 8)
        inits = {t.name for t in m.graph.initializer}
        keep = [i for i in m.graph.input if i.name not in inits]
        del m.graph.input[:]
        m.graph.input.extend(keep)
        f13 = os.path.join(tmp, "opset13.onnx")
        onnx.save(m, f13)
        pre = os.path.join(tmp, "pre.onnx")
        quant_pre_process(f13, pre)
        m = explicit_reshapes(onnx.load(pre))
        if any(n.op_type == "BatchNormalization" for n in m.graph.node):
            sys.exit("BatchNormalization left unfolded: the NPU would get "
                     "it wrong")
        onnx.save(m, pre)
        inp = m.graph.input[0]
        size = inp.type.tensor_type.shape.dim[-1].dim_value
        quantize_static(pre, dst, Reader(inp.name, size, images),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8)
    print("wrote", dst)


if __name__ == "__main__":
    main()
