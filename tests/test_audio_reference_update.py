"""Regression checks for scoped native dispatch and the v0.1.0 workflow contract."""
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fish_token_guard_test', ROOT / 'h3_token_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def workflow_contract(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SelfLiftH3Sampler')
    schema = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'INPUT_TYPES')
    sample = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'sample')
    namespace = {'_upscaler_input': lambda: ('UPSCALE_MODELS_FROM_DISK', {})}
    exec(compile(ast.Module(body=[schema], type_ignores=[]), '<schema>', 'exec'), namespace)
    inputs = namespace['INPUT_TYPES'].__func__(None)
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k != 'tooltip'}
        if isinstance(value, (tuple, list)):
            return [strip(v) for v in value]
        return value
    return {'inputs': strip(inputs), 'input_order': {k: list(v) for k, v in inputs.items()},
            'sample_args': [arg.arg for arg in sample.args.args],
            'sample_defaults': [ast.unparse(d) for d in sample.args.defaults]}


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.original = guard.ck.sol_attn_chunked
        guard._DISPATCH = None
        guard._LOGGED = False
        guard._WARNED.clear()
        guard._function_for_device.cache_clear()
        guard.ck.sol_attn_chunked = lambda *a, **kw: 'native'
        self.rope = SimpleNamespace(device=SimpleNamespace(type='cuda', index=1))

    def tearDown(self):
        guard.ck.sol_attn_chunked = self.original
        guard._DISPATCH = None
        guard._function_for_device.cache_clear()

    def invoke(self, tokens=256):
        return guard.ck.sol_attn_chunked(None, None, None, self.rope, token_aug=tokens)

    @patch.object(guard, 'version', return_value='0.2.40')
    @patch.object(guard, '_function_for_device', return_value=lambda *a, **kw: 'fixed')
    def test_scope_thread_device_and_zero_augmentation(self, select, version):
        with guard.corrected_sparse_tokens():
            self.assertEqual(self.invoke(), 'fixed')
            select.assert_called_once_with(1)
            self.assertEqual(self.invoke(0), 'native')
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(self.invoke).result(), 'native')
        self.assertEqual(self.invoke(), 'native')

    @patch.object(guard, 'version', return_value='0.2.40')
    @patch.object(guard, '_function_for_device', return_value=lambda *a, **kw: 'fixed')
    def test_keyword_and_positional_arguments(self, select, version):
        with guard.corrected_sparse_tokens():
            self.assertEqual(guard.ck.sol_attn_chunked(rope_freqs=self.rope, token_aug=256), 'fixed')
            args = [None] * 17
            args[3], args[16] = self.rope, 256
            self.assertEqual(guard.ck.sol_attn_chunked(*args), 'fixed')

    @patch.object(guard, 'version', return_value='0.2.40')
    @patch.object(guard, '_function_for_device', return_value=lambda *a, **kw: 'fixed')
    def test_nested_scope_restored_on_failure(self, select, version):
        with guard.corrected_sparse_tokens():
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with guard.corrected_sparse_tokens():
                    raise RuntimeError('test failure')
            self.assertEqual(self.invoke(), 'fixed')
        self.assertEqual(self.invoke(), 'native')

    @patch.object(guard, 'version', return_value='0.2.41')
    @patch.object(guard, '_function_for_device', side_effect=AssertionError('wrong ABI'))
    def test_unknown_abi_falls_back_with_one_notice(self, select, version):
        with self.assertLogs(level='WARNING') as logs, guard.corrected_sparse_tokens():
            self.assertEqual(self.invoke(), 'native')
            self.assertEqual(self.invoke(), 'native')
        self.assertEqual(len(logs.output), 1)
        select.assert_not_called()

    @patch.object(guard, 'version', return_value='0.2.40')
    @patch.object(guard, '_function_for_device', return_value=None)
    def test_unavailable_optional_core_retains_native(self, select, version):
        with guard.corrected_sparse_tokens():
            self.assertEqual(self.invoke(), 'native')

    @patch.object(guard, 'version', return_value='0.2.40')
    def test_cuda_execution_failure_is_not_silently_retried(self, version):
        def failed(*a, **kw):
            raise RuntimeError('CUDA execution failed')
        with patch.object(guard, '_function_for_device', return_value=failed):
            with guard.corrected_sparse_tokens(), self.assertRaisesRegex(RuntimeError, 'CUDA execution'):
                self.invoke()
        self.assertEqual(self.invoke(), 'native')


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        guard._function_for_device.cache_clear()
        guard._WARNED.clear()

    def tearDown(self):
        guard._function_for_device.cache_clear()

    @patch.object(guard.sys, 'platform', 'linux')
    @patch.object(guard, '_corrected_function', side_effect=AssertionError('must not load'))
    def test_other_os_preserves_native(self, load):
        self.assertIsNone(guard._function_for_device(0))

    @patch.object(guard.sys, 'platform', 'win32')
    @patch.object(guard.torch.cuda, 'get_device_capability', return_value=(8, 9))
    @patch.object(guard, '_corrected_function', side_effect=AssertionError('must not load'))
    def test_other_gpu_preserves_native(self, load, capability):
        self.assertIsNone(guard._function_for_device(2))
        capability.assert_called_once_with(2)

    @patch.object(guard.sys, 'platform', 'win32')
    @patch.object(guard.torch.cuda, 'get_device_capability', return_value=(12, 0))
    @patch.object(guard, '_driver_version', return_value=13000)
    @patch.object(guard, '_corrected_function', side_effect=AssertionError('must not load'))
    def test_older_driver_preserves_native(self, load, driver, capability):
        self.assertIsNone(guard._function_for_device(0))

    @patch.object(guard.sys, 'platform', 'win32')
    @patch.object(guard.torch.cuda, 'get_device_capability', return_value=(12, 0))
    @patch.object(guard, '_driver_version', return_value=13030)
    def test_missing_binary_or_dependency_preserves_native(self, driver, capability):
        for error in (FileNotFoundError('missing'), OSError('dependency missing'), AttributeError('missing export')):
            guard._function_for_device.cache_clear()
            with patch.object(guard, '_corrected_function', side_effect=error):
                self.assertIsNone(guard._function_for_device(0))

    @patch.object(guard.sys, 'platform', 'win32')
    @patch.object(guard.torch.cuda, 'get_device_capability', return_value=(12, 0))
    @patch.object(guard, '_driver_version', return_value=13030)
    @patch.object(guard, '_corrected_function', return_value='verified core')
    def test_supported_core_selected_once_per_device(self, load, driver, capability):
        self.assertEqual(guard._function_for_device(1), 'verified core')
        self.assertEqual(guard._function_for_device(1), 'verified core')
        load.assert_called_once()
        capability.assert_called_once_with(1)


class WorkflowTests(unittest.TestCase):
    def test_v010_inputs_order_defaults_and_call_signature(self):
        expected = json.loads((ROOT / 'tests/fixtures/h3_v010_contract.json').read_text())
        self.assertEqual(workflow_contract((ROOT / 'nodes.py').read_text(encoding='utf-8')), expected)

    def test_reference_detection_does_not_add_steps_for_images(self):
        tree = ast.parse((ROOT / 'nodes.py').read_text(encoding='utf-8'))
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_has_reference_audio')
        ns = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), '<audio detection>', 'exec'), ns)
        detect = ns['_has_reference_audio']
        self.assertFalse(detect([[None, {}]]))
        self.assertFalse(detect([[None, {'minimax_refs': [{'kind': 'image', 'latent': object()}]}]]))
        self.assertFalse(detect([[None, {'minimax_refs': [{'audio_latent': None}]}]]))
        self.assertTrue(detect([[None, {'minimax_refs': [{'audio_latent': object()}]}]]))
        self.assertTrue(detect([[None, {'minimax_refs': [{'kind': 'video_audio', 'audio_latent': object()}]}]]))


if __name__ == '__main__':
    unittest.main()
