"""The stage boundary must only retire completed, thread-local scratch."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fish_stage_workspace', ROOT / 'h3_memory.py')
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)


class StageWorkspaceTests(unittest.TestCase):
    def invoke(self, graph, device='cuda:0', other=None):
        key = threading.get_ident()
        graphs = {key: graph} if graph is not None else {}
        if other is not None:
            graphs[key + 1] = other
        with patch.object(memory.comfy.model_prefetch, 'MALLOC_GRAPHS', graphs), \
             patch.object(memory.comfy.model_prefetch, 'cleanup_malloc_graph') as cleanup:
            result = memory.release_stage_workspace(device)
            return result, cleanup.call_count, graphs

    def test_completed_large_graph_is_retired(self):
        graph = SimpleNamespace(_comfy_active=False, physical_bytes=6 * 1024**3)
        other = SimpleNamespace(_comfy_active=True, physical_bytes=8 * 1024**3)
        result, calls, graphs = self.invoke(graph, other=other)
        self.assertTrue(result)
        self.assertEqual(calls, 1)
        self.assertIs(graphs[threading.get_ident()+1], other)

    def test_active_graph_is_never_retired(self):
        graph = SimpleNamespace(_comfy_active=True, physical_bytes=6 * 1024**3)
        self.assertEqual(self.invoke(graph)[:2], (False, 0))

    def test_unknown_scope_is_not_assumed_safe(self):
        self.assertEqual(self.invoke(SimpleNamespace(physical_bytes=6 * 1024**3))[:2], (False, 0))

    def test_cpu_and_missing_or_small_graph_are_noops(self):
        graph = SimpleNamespace(_comfy_active=False, physical_bytes=6 * 1024**3)
        self.assertEqual(self.invoke(graph, 'cpu')[:2], (False, 0))
        self.assertEqual(self.invoke(None)[:2], (False, 0))
        graph.physical_bytes = 24 * 1024**2
        self.assertEqual(self.invoke(graph)[:2], (False, 0))

    def test_older_comfy_without_graph_api_is_noop(self):
        with patch.object(memory.comfy.model_prefetch, 'MALLOC_GRAPHS', None):
            self.assertFalse(memory.release_stage_workspace('cuda:0'))


class WorkspacePressureTests(unittest.TestCase):
    def trim(self, reserved, allocated, free, device='cuda:0'):
        gib = 1024**3
        with patch.object(memory.torch.cuda, 'memory_reserved', return_value=int(reserved*gib)), \
             patch.object(memory.torch.cuda, 'memory_allocated', return_value=int(allocated*gib)), \
             patch.object(memory.torch.cuda, 'get_device_properties', return_value=SimpleNamespace(total_memory=16*gib)), \
             patch.object(memory.torch.cuda, 'mem_get_info', return_value=(int(free*gib),16*gib)), \
             patch.object(memory.torch.cuda, 'device', return_value=nullcontext()), \
             patch.object(memory.comfy.model_prefetch, 'pause_malloc_graph', return_value=nullcontext()), \
             patch.object(memory.torch.cuda, 'empty_cache') as empty:
            memory.trim_workspace_cache(device)
            return empty.call_count

    def test_external_graph_pressure_below_old_threshold(self):
        # Reproduced 16 GB GPU: only 5 GB counted by PyTorch, but no free VRAM.
        self.assertEqual(self.trim(5.06, .73, 0), 1)

    def test_same_pool_with_headroom_is_kept(self):
        self.assertEqual(self.trim(5.06, .73, 5), 0)

    def test_live_allocations_are_not_treated_as_reclaimable(self):
        self.assertEqual(self.trim(5.06, 4.9, 0), 0)

    def test_large_inactive_pool_preserves_previous_policy(self):
        self.assertEqual(self.trim(8, 1, 5), 1)

    def test_cpu_does_not_touch_cuda_allocator(self):
        self.assertEqual(self.trim(8, 1, 0, 'cpu'), 0)


if __name__ == '__main__':
    unittest.main()
