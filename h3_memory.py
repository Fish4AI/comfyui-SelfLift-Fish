"""Release completed allocation plans when the sampling geometry changes."""
import logging
import threading

import torch
import comfy.model_prefetch


def trim_workspace_cache(device):
    """Release unused PyTorch scratch before a streamed model prediction.

    Preview/upscaler allocations can coexist with DynamicVRAM graph pages and
    weight storage that are not represented in PyTorch's reserved-byte count.
    Therefore a fraction-of-VRAM threshold alone misses real device pressure.
    Keep live tensors and caches; reclaim only inactive allocator storage.
    """
    device = torch.device(device)
    if device.type != 'cuda':
        return
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    inactive = reserved - allocated
    if inactive <= 512 * 1024**2:
        return
    total = torch.cuda.get_device_properties(device).total_memory
    large_pool = reserved > max(2 * 1024**3, total * .4) and inactive > 2 * 1024**3
    free, _ = torch.cuda.mem_get_info(device)
    device_pressure = free < max(1024**3, total * .1)
    if not (large_pool or device_pressure):
        return
    with torch.cuda.device(device), comfy.model_prefetch.pause_malloc_graph():
        torch.cuda.empty_cache()
    logging.info('[SelfLift memory] inactive workspace trimmed: reserved %.2f -> '
                 '%.2f GiB; live tensors and attention caches retained',
                 reserved / 1024**3, torch.cuda.memory_reserved(device) / 1024**3)


def release_stage_workspace(device):
    """Retire the current thread's inactive DynamicVRAM scratch graph.

    H3 ends its allocation-graph scope after each prediction but retains its
    physical pages and allocation plan. SelfLift changes both token geometry
    and the projection schedule at the low/high boundary. Carrying the old
    plan into that different workload can retain gigabytes of unnecessary
    scratch under cudaMallocAsync. Native graph cleanup synchronizes pending
    work and releases graph-owned scratch, without unloading model weights or
    clearing the sparse attention statistics allocated outside that graph.

    Only call between sampling stages, after the model has returned its
    independent outputs. Never retire a graph while a forward is active.
    """
    if torch.device(device).type != 'cuda':
        return False
    prefetch = comfy.model_prefetch
    graphs = getattr(prefetch, 'MALLOC_GRAPHS', None)
    cleanup = getattr(prefetch, 'cleanup_malloc_graph', None)
    if graphs is None or cleanup is None:
        return False
    graph = graphs.get(threading.get_ident())
    if graph is None or getattr(graph, '_comfy_active', True):
        return False
    physical = getattr(graph, 'physical_bytes', 0)
    if physical < 512 * 1024**2:
        return False
    cleanup()
    logging.info('[SelfLift memory] resolution switch: retired %.2f GiB of '
                 'completed allocation-graph workspace; model weights and '
                 'attention statistics retained', physical / 1024**3)
    return True
