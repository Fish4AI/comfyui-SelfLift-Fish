# Measurements, scope and reproduction

## Reference-image update (2026-10-30)

Compared the previously installed SelfLift-Fish with this update, not unmodified facok upstream. RTX 5070 Ti 16 GB / 128 GB RAM; two matching-resolution reference images, 1376×768, 124 frames at 24 fps, seed 82341234. Native INT8 ref2va, DMAD v4 LoRA 0.8, Euler/beta 10 steps, transition 6, lowres scale 0.7, CFG 1; Global stream, chunk 4096, weight cache 4 GiB, SLA 10%, extra_tokens=256.

Interleaved order: updated, baseline, updated, baseline. Means of two complete video renders per version:

| Measurement | Before | Updated | Time reduction |
| --- | ---: | ---: | ---: |
| Low-resolution stage | 57.507 s | 55.185 s | 4.0% |
| High-resolution stage | 51.242 s | 47.980 s | 6.4% |
| Sampling including upscale | 111.127 s | 105.601 s | 5.0% |
| Sampling, AV decoding and video export | 124.567 s | 119.087 s | 4.4% |

Total runs: before 124.533 / 124.601 s; updated 120.128 / 118.046 s. Condition encoding was cached; timing excludes initial text/reference encoding and model loading. Preview MP4s contain video only; decoded audio was compared separately. This is not a cold-start benchmark or a measured speedup for 20-second/2 MP/FP8 generation.

The update reuses inputs within one layer/forward and prepares weights within existing low-stage reference blocks. It does not cache reference features across denoising steps or change reference dimensions, steps or sparse settings. Temporary input cache is capped at 256 MiB with a free-memory guard.

Quality: native last-layer attention with extra_tokens=256 differed even when the original code repeated an identical input. In a separate deterministic complete-video comparison, both sides used extra_tokens=0: both stage latents, all 124 decoded frames and decoded audio matched bit for bit. Production code and performance runs retain extra_tokens=256. This validates the tested configuration, not universal bitwise equality. BF16, INT8 and scaled FP8 layer regressions also passed.

PyTorch allocated-tensor peaks averaged 2.577 GiB before and 2.857 GiB after; these exclude DynamicVRAM mappings and are not whole-GPU peaks. No whole-GPU peak reduction is claimed. Node IDs, interfaces and saved parameter ordering are unchanged.

---

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
successful 1.0 MP runs above were checked for execution success, not independently
scored for background quality. A paired visual benchmark is not bundled.

No complete-run VRAM peak is available for these latest real-video runs. The
16 GB GPU capacity is known; the isolated allocator measurements below must not
be substituted for whole-workflow peaks. The system had 128 GB host RAM and
used DynamicVRAM offloading; smaller-RAM systems were not validated.

### A2. 2.0 MP high-resolution results

The same machine also completed a **15-second, 1920 × 1088** workflow after the
stream and upscaler fixes. The log records 107 temporal latent positions,
target latent grid 68 × 120 and 219709 packed tokens. Configuration: ten Euler
evaluations split 6/4, scale 0.60, cfg 1, `rho=0`, external H3 upscaler,
native sparse attention, Global stream, chunk 4096, weight budget 4 GiB.

| Measurement | Earlier global, 10 s | Optimized stream before upscaler fix, 10 s | Stream + upscaler fix, 15 s |
| --- | ---: | ---: | ---: |
| Target pixels | 1920 × 1088 | 1920 × 1088 | 1920 × 1088 |
| Temporal latent positions | 72 | 72 | 107 |
| Whole workflow | 1670 s (27m 50s) | 775 s (12m 55s) | 848 s (14m 08s) |
| Complete `paired_lifts` | not separately recorded here | 281.854 s | 10.408 s |
| Target-resolution step 1 | 209.450 s | 75.147 s | 157.913 s |
| Target-resolution step 2 | 732.072 s | 60.351 s | 101.975 s |
| Target-resolution step 3 | 188.878 s | 55.895 s | 112.816 s |
| Target-resolution step 4 | 189.378 s | 55.888 s | 100.402 s |
| Target-resolution stage total | 1322.042 s | 249.757 s | 477.195 s |

For the two **10-second** records, the observed complete workflow dropped
53.6% (1670/775 ≈ 2.15× recorded throughput), and the target-resolution stage
dropped 81.1% (1322.042/249.757 ≈ 5.29×). These are development-run observations
at the same duration and target dimensions, not a paired same-prompt,
same-seed benchmark or a comparison with facok's unmodified spatial tiling.
Prompt/reference token packing, caching, load state and runtime conditions can
contribute. Do not use the different-length 15-second run to calculate those
ratios or compare its 10.408-second lift directly with the shorter run's lift
as an isolated-kernel speedup.

Sources: the earlier collector result and execution logs on 2026-10-19,
07:38:27–07:51:12 for the 10-second stream run, and
10:07:30–10:21:34 for the 15-second updated run. The latter logged the complete
ten-step sampling, VAE loading and `Prompt executed in 00:14:08`; no full-run
resource collector was active. These do not establish a complete-run VRAM peak.

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
