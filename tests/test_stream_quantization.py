"""Run with ComfyUI's Python: test_stream_quantization.py --comfy-root PATH.

Uses small CUDA tensors and the installed ComfyUI/Kitchen operators. No models
are downloaded and no ComfyUI server or generation workflow is started.
"""
import argparse
import dataclasses
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument('--comfy-root', required=True)
config, unittest_args = parser.parse_known_args()
sys.path.insert(0, config.comfy_root)
sys.argv = [sys.argv[0]]

import torch
import comfy.ops
from comfy.quant_ops import QuantizedTensor

package = types.ModuleType('selflift_quantization_tests')
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
stream = importlib.import_module(f'{package.__name__}.h3_stream_attention')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for native quantized GEMM')
class StreamQuantizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8234)
        self.x = torch.randn(257, 64, device='cuda', dtype=torch.bfloat16)
        # Ten heads deliberately leaves a partial group after the first eight.
        self.w = torch.randn(3 * 10 * 128, 64, device='cuda', dtype=torch.bfloat16) * .05
        self.bias = torch.randn(self.w.shape[0], device='cuda', dtype=torch.bfloat16) * .01

    def grouped(self, weight, input_scale=None):
        output = torch.empty(len(self.x), len(self.w), device='cuda', dtype=self.x.dtype)
        # Fail on silent full-weight/group dequantization: the FP8 acceleration
        # must remain active, not merely pass the validation guard.
        with patch.object(QuantizedTensor, 'dequantize', side_effect=AssertionError('dequantization fallback')):
            for first, last in ((0, 8), (8, 10)):
                indices = torch.arange(first * 128, last * 128, device='cuda')
                rows = torch.cat([indices + i * 10 * 128 for i in range(3)])
                selected = stream.select_weight_rows(weight, rows)
                self.assertIsInstance(selected, QuantizedTensor)
                self.assertEqual(selected._qdata.dtype, weight._qdata.dtype)
                for a in range(0, len(self.x), 64):
                    output[a:a+64, rows] = stream.project_qkv(
                        self.x[a:a+64], selected, self.bias[rows], input_scale)
        return output

    @torch.inference_mode()
    def test_fp8_matches_native_with_saved_and_default_input_scales(self):
        ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
        for scale in (None, torch.tensor(.03125, device='cuda')):
            with self.subTest(input_scale=scale):
                weight = QuantizedTensor.from_float(self.w, 'TensorCoreFP8E4M3Layout', scale='recalculate')
                layer = ops.Linear(64, len(self.w), bias=True, device='cuda')
                layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                layer.bias = torch.nn.Parameter(self.bias, requires_grad=False)
                layer.layout_type = weight._layout_cls
                layer.quant_format = 'float8_e4m3fn'
                if scale is not None:
                    layer.input_scale = scale
                stream.validate_projection(layer)
                expected = layer(self.x)
                actual = self.grouped(weight, scale)
                torch.testing.assert_close(actual, expected, rtol=.01, atol=.01)

    @torch.inference_mode()
    def test_int8_scalar_and_row_scales_remain_native(self):
        for per_channel in (False, True):
            with self.subTest(per_channel=per_channel):
                weight = QuantizedTensor.from_float(self.w, 'TensorWiseINT8Layout', per_channel=per_channel)
                expected = torch.nn.functional.linear(self.x, weight, self.bias)
                actual = self.grouped(weight)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_fp8_rows_preserve_storage_values_and_scale(self):
        weight = QuantizedTensor.from_float(self.w, 'TensorCoreFP8E4M3Layout', scale='recalculate')
        rows = torch.tensor([0, 200, 1500, 3000], device='cuda')
        with patch.object(QuantizedTensor, 'dequantize', side_effect=AssertionError('dequantization fallback')):
            selected = stream.select_weight_rows(weight, rows)
        self.assertEqual(selected._params.scale.data_ptr(), weight._params.scale.data_ptr())
        torch.testing.assert_close(selected._qdata.float(), weight._qdata.float()[rows], rtol=0, atol=0)

    @torch.inference_mode()
    def test_invalid_scale_geometry_is_rejected_before_sampling(self):
        weight = QuantizedTensor.from_float(self.w, 'TensorCoreFP8E4M3Layout', scale='recalculate')
        bad = QuantizedTensor(weight._qdata, weight._layout_cls,
                              dataclasses.replace(weight._params, scale=torch.ones(len(self.w), 1, device='cuda')))
        with self.assertRaisesRegex(ValueError, 'scale geometry'):
            stream.validate_projection(types.SimpleNamespace(weight=bad))


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *unittest_args])
