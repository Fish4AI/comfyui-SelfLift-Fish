"""Scoped compatibility fix for comfy-kitchen 0.2.40 token-bin overflow.

The native producer, statistics, sinks, budget and sparse layout are retained.
Only the CUDA core's saturated histogram cutoff is corrected. A ContextVar
keeps other nodes/threads on their original backend. The bundled core supports
Windows sm_120; unsupported environments retain native attention with a notice.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import ctypes
import logging
import sys
import threading
import types

import torch
import comfy_kitchen as ck

_ACTIVE = ContextVar('selflift_token_guard_active', default=False)
_LOCK = threading.Lock()
_LOAD_LOCK = threading.Lock()
_DISPATCH = None
_FIXED = None
_LIBRARY = None
_LOGGED = False
_WARNED = set()


def _warn_once(reason):
    with _LOCK:
        if reason in _WARNED:
            return
        _WARNED.add(reason)
    logging.warning('[SelfLift token guard] %s; using native attention without '
                    'the token-budget correction. Audio-reference transition '
                    'handling is independent of this optional kernel.', reason)


def _driver_version():
    driver = ctypes.WinDLL('nvcuda.dll')
    query = driver.cuDriverGetVersion
    query.argtypes = [ctypes.POINTER(ctypes.c_int)]
    query.restype = ctypes.c_int
    value = ctypes.c_int()
    status = query(ctypes.byref(value))
    if status:
        raise OSError(f'cuDriverGetVersion returned {status}')
    return value.value


@lru_cache(maxsize=16)
def _function_for_device(device_index):
    # Check before loading or launching anything from the bundled CUDA binary.
    if sys.platform != 'win32' or ctypes.sizeof(ctypes.c_void_p) != 8:
        _warn_once('the bundled core requires Windows x64')
        return None
    if torch.cuda.get_device_capability(device_index) != (12, 0):
        _warn_once('the bundled core requires an sm_120 GPU')
        return None
    try:
        if _driver_version() < 13010:
            _warn_once('the bundled core requires a CUDA 13.1-capable driver')
            return None
        with _LOAD_LOCK:
            return _corrected_function()
    except (OSError, ImportError, AttributeError) as error:
        # These are initialization failures, before any kernel launch. Never
        # retry a CUDA execution failure on another backend in the same context.
        _warn_once(f'optional core could not be loaded ({type(error).__name__})')
        return None


def _corrected_function():
    global _FIXED, _LIBRARY
    if _FIXED is not None:
        return _FIXED
    import comfy_kitchen.backends.cuda as cuda
    path = Path(__file__).parent / 'native/token_guard/core_fixed.dll'
    if not path.is_file():
        raise FileNotFoundError(path)
    lib = ctypes.CDLL(str(path))
    core = lib.fish_core
    core.argtypes = ([ctypes.c_void_p] * 7 + [ctypes.c_int] * 2
                     + [ctypes.c_float] * 2 + [ctypes.c_int] * 6 + [ctypes.c_void_p])
    core.restype = ctypes.c_int

    class NativeCore:
        def __getattr__(self, name):
            return getattr(cuda._C, name)

        def sol_attn_core(self, ws, out, vscale, kmean, vamax, batch, t, h,
                          tau, scale, ss, se, sqs, sqe, stream, *, threshold=None,
                          block_len=None, tail=True, token_aug=0):
            if batch != 1:
                raise ValueError('SelfLift native token guard expects the H3 packed batch of one.')
            # Capsules are already exported on the caller's CUDA stream.
            tensors = [torch.from_dlpack(x) if x is not None else None
                       for x in (ws, out, vscale, kmean, vamax, threshold, block_len)]
            pointers = [x.data_ptr() if x is not None else None for x in tensors]
            status = core(*pointers, t, h, tau, scale, ss, se, sqs, sqe,
                          int(tail), int(token_aug), stream)
            if status:
                raise RuntimeError(f'SelfLift token guard CUDA error: {status}')

    namespace = dict(cuda.sol_attn_chunked.__globals__)
    namespace['_C'] = NativeCore()
    fn = types.FunctionType(cuda.sol_attn_chunked.__code__, namespace,
                            argdefs=cuda.sol_attn_chunked.__defaults__)
    fn.__kwdefaults__ = cuda.sol_attn_chunked.__kwdefaults__
    _LIBRARY, _FIXED = lib, fn
    return fn


def _install_dispatch():
    global _DISPATCH
    with _LOCK:
        if _DISPATCH is not None:
            return
        # The private workspace ABI is version-specific. New versions must be
        # evaluated independently; never pass a changed workspace to this core.
        original = getattr(ck, 'sol_attn_chunked', None)
        if original is None:
            return
        try:
            kitchen_version = version('comfy-kitchen')
        except PackageNotFoundError:
            kitchen_version = 'unknown'

        @wraps(original)
        def dispatch(*args, **kwargs):
            global _LOGGED
            # Native callers use keyword token_aug; support its positional form
            # too without changing the public comfy-kitchen signature.
            token_aug = kwargs.get('token_aug', args[16] if len(args) > 16 else 0)
            rope = args[3] if len(args) > 3 else kwargs.get('rope_freqs')
            if not (_ACTIVE.get() and token_aug > 0 and rope is not None
                    and rope.device.type == 'cuda'):
                return original(*args, **kwargs)
            if kitchen_version != '0.2.40':
                _warn_once(f'comfy-kitchen {kitchen_version} has not been verified; expected 0.2.40')
                return original(*args, **kwargs)
            device_index = rope.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            fixed = _function_for_device(device_index)
            if fixed is None:
                return original(*args, **kwargs)
            if not _LOGGED:
                logging.info('[SelfLift token guard] corrected saturated histogram-bin selection; native sparse budget and full context retained')
                _LOGGED = True
            return fixed(*args, **kwargs)

        _DISPATCH = dispatch
        ck.sol_attn_chunked = dispatch


@contextmanager
def corrected_sparse_tokens():
    _install_dispatch()
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
