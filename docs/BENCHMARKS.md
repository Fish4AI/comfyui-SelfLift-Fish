# Measurements, scope and reproduction

Recorded 2026-10-18/19 on an RTX 5070 Ti 16 GB and 128 GB system RAM.
The tested CUDA stack for the upscaler investigation was PyTorch
2.9.1+cu130, with ComfyUI flagging the affected cuDNN memory-workaround range.
These are local engineering measurements, not a cross-hardware benchmark suite.

## A. Complete real-video workflows

Native INT8 H3, LoRA patches enabled, Euler / beta 10 steps, low-resolution
steps 6, scale 0.60, cfg 1, `rho=0`, learned H3 latent upscaler, native SLA 10%,
Global stream, token chunk 4096, weight cache 4 GiB. Runs used different content.

| Requested duration | Target | Whole workflow seconds | High-resolution step seconds | History status |
| --- | --- | ---: | --- | --- |
| 15 s | 1376 × 768 | 329.338 | 47.567 / 37.077 / 37.079 / 37.213 | Success |
| 15 s | 1376 × 768 | 331.653 | 52.102 / 41.052 / 41.094 / 41.075 | Success |
| 20 s | 1376 × 768 | 492.147 | 74.620 / 67.396 / 61.475 / 61.181 | Success |
| 20 s | 1376 × 768 | 469.373 | 69.081 / 65.565 / 60.448 / 54.232 | Success |
| 15 s | 1376 × 768 | 327.492 | 50.017 / 40.044 / 39.983 / 40.169 | Success |
| 12 s | 1376 × 768 | 263.246 | 38.858 / 31.979 / 30.808 / 30.661 | Success |

Source: local ComfyUI execution history start/success timestamps, checked
against corresponding `Prompt executed` and per-step timing log entries.
The first 20-second run logged `transition paired_lifts: 7.040s`.
Requested duration is the workflow control, not a promise of exact container
duration after frame rounding. Whole-workflow times include the executed graph
and reflect its cache/load state; they do not include time waiting in a queue.

The owner's earlier controlled visual comparison confirmed that the global
path resolved the reproduced spatial-tiling background deformation. The six
successful runs above were checked for execution success, not independently
scored for background quality. A paired visual benchmark is not bundled.

No complete-run VRAM peak is available for these latest six runs. The 16 GB GPU
capacity is known; the isolated allocator measurements below must not be
substituted for whole-workflow peaks. The system had 128 GB host RAM and used
DynamicVRAM offloading; smaller-RAM systems were not validated.

## B. Same-checkpoint upscaler comparison

Real `h3_upscaler_lms_v0.1.safetensors`, FP16, batch 1, 24 latent channels.
Input `(1,24,107,34,62)` → output `(1,24,107,58,104)`.
Both paths retain temporal chunk 32, overlap 5, including the same 47/52-frame
effective windows. No video generation was submitted for this isolated test.

| Metric | Previous network | Guarded Conv3d path |
| --- | ---: | ---: |
| Network wall time | 74.8336 s | 7.4739 s |
| Peak PyTorch allocated | 10.0046 GiB | 2.5395 GiB |
| Peak PyTorch reserved | 21.5371 GiB | 4.8887 GiB |

Network speedup: about 10.01×. Peak allocated reduction: about 74.6%; peak
reserved reduction: about 77.3%. Reserved above physical GPU capacity is an
allocator observation on the tested Windows environment, not a claim of that
much dedicated VRAM on the card. These counters exclude memory managed outside
the PyTorch allocator.

Complete optimized `learned_latent_lift`, including DynamicVRAM load,
normalization, GPU compute, CPU return and unload: 7.919 s, allocated peak
2.539 GiB / reserved peak 4.900 GiB.

Second shape `(1,24,72,40,72)` → `(1,24,72,68,120)`: complete optimized lift
7.133 s, allocated 3.169 GiB / reserved 6.324 GiB. This second shape has no
paired baseline reported here.

### Numerical validation

Outputs were finite; the actual checkpoint loaded strictly. Compared with the
previous FP16 network, relative L2 difference was about 0.000964 (0.0964%), mean
absolute difference 0.002182 and maximum absolute difference 0.125. Different
convolution kernels can round differently; output is not claimed bitwise equal.
The optimized output matched the tested channels-last-3D comparison path
bitwise. CPU FP16/FP32, CUDA FP32 and ordinary/depthwise fallback checks passed.

## C. Earlier real-video prototypes — historical context

These are development revisions of the Fish implementation, **not a benchmark
against facok's unmodified spatial sampler**. Same broad 10-second 1920 × 1088
workflow (149080 tokens, 56 heads, SLA 10%), with implementation/chunk changes
listed below and differing runtime/cache state.

| Build | Mode | Token chunk | Whole workflow | High-resolution step seconds |
| --- | --- | ---: | --- | --- |
| Early global | global | 1024 | 27m 50s | 209.450 / 732.072 / 188.878 / 189.378 |
| RAM prototype | global_ram | 1024 | 53m 48s | 442.109 / 1140.741 / 729.815 / 388.473 |
| Initial stream prototype | global_stream | 4096 | 59m 00s | 73.134 / 1007.181 / 1075.701 / 1019.654 |

All completed. RAM staging saved a local tensor allocation but made this real
workflow slower. The first stream prototype still omitted full input/output
intermediates and suffered memory-pool/weight-residency competition. Those
results explain why the final implementation bounds the entire forward path
instead of relying on a single-layer benchmark.

Later same-model 149080-token continuous full-forward diagnostics, with original
tiny-VAE previews, recorded 64.21 / 52.35 / 52.23 seconds for three forwards.
They used diagnostic inputs and omit real conditioning, the full low-resolution
prefix, upscaling, Euler updates, final VAE and saving. They support the specific
regression investigation, **not an end-to-end generation speedup factor**.

Historical RAM/stream resource collectors covered only part of each run.
Their observed peaks are not complete-run peaks. Shared GPU memory includes
mapped/staged allocations and is not a direct measurement of paging or PCIe
traffic.

## Reproducing a comparison

1. Record GPU, VRAM, host RAM, driver, PyTorch/CUDA/cuDNN, ComfyUI revision,
   model/LoRA identities and weights, backend, previews and cache state.
2. Fix prompt, reference assets, seed, output dimensions, frame count, sigma
   schedule and all sparse-attention settings. Change only the tested control.
3. Preserve full step/stage logs and use ComfyUI execution history for the
   complete graph time. Distinguish cached runs from recomputed graphs.
4. Monitor GPU and host resources over the whole interval if reporting peaks.
   Separate allocated, reserved, device-wide dedicated and shared counters.
5. Check the resulting video for scene geometry, background motion, identity,
   textures and audio synchronization. Successful execution alone is not a
   visual-quality score.
