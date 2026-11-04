# Native token budget guard

This folder contains a minimal compatibility core for **comfy-kitchen 0.2.40**.
Original CUDA sources come from [Comfy-Org/comfy-kitchen v0.2.40](https://github.com/Comfy-Org/comfy-kitchen/tree/v0.2.40/comfy_kitchen/backends/cuda/sage_attention), under Apache-2.0; copyright/license headers are retained.

Only two selection expressions in `sol_attn_token.cu` change: keep the cutoff as a histogram-bin index and compare candidate bins directly. The saturated top bin must be rejected as a whole when it exceeds the token budget. Converting that bin's upper edge into a finite score lets an oversized set compete for a capped atomic append list.

`core_wrapper.cu` exposes the unchanged core ABI to the process-local Python guard. The native QKV producer stays in the installed package. The guard runs only within SelfLift's sampling context; ordinary ComfyUI callers remain on their original function.

The included `core_fixed.dll` is a **Windows x64 / sm_120 / CUDA 13.1 build**, verified on RTX 5070 Ti with PyTorch 2.9.1+cu130 and a CUDA-13.3-capable driver. Its SHA256 is `61e1c9cc07dd41417fdf3201c29119531a56b0d5b338b346ef39455295e69aee`.

The Python dispatcher checks kitchen version, OS, device architecture and driver support before loading the binary. It requires a driver reporting CUDA 13.1 or newer. Unsupported configurations and missing/unloadable binaries fall back to native attention with a notice; this does **not** apply the token correction there. The sampler's audio-reference transition remains independent. No installed comfy-kitchen package is overwritten and no automatic build runs at import.

Build (from a configured CUDA/MSVC developer shell):

```text
nvcc --shared -O3 --use_fast_math -arch=sm_120 -Xcompiler /wd4819 sol_attn.cu sol_attn_route.cu sol_attn_token.cu sol_attn_exact.cu sol_attn_producer.cu sol_attn_preprocess.cu sol_attn_vtranspose.cu core_wrapper.cu -o core_fixed.dll
```

Use CUDA 13.1 and MSVC 14.44 (VS 2022) to reproduce the tested build; pass `-ccbin` if MSVC is not on PATH. The checked-in binary was compiled before the explanatory modification-notice comment was added to `sol_attn_token.cu`; executable source is unchanged. Rebuilding for another architecture requires separate validation and an explicit dispatcher update, not just changing the compiler flag.

See [full-video results](../../docs/AUDIO_REFERENCE_UPDATE.md) for the tested scope. This repair does not disable sparse attention, remove references, or freeze/composite video frames. License text is included in [LICENSE](LICENSE).
