# selflift-Fish

**Global-context memory saving for MiniMax H3 in ComfyUI — stable backgrounds, native sparse attention, and practical long-video generation on a 16 GB GPU.**

[中文说明](README_CN.md) · [Benchmarks](docs/BENCHMARKS.md) · [Settings](docs/SETTINGS.md) · [Attribution](NOTICE.md)

Based on [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift). This project builds on its progressive-resolution sampler and H3 integration, with a focus on the memory and performance problems encountered when generating longer, higher-resolution videos.

## What improves

- **Fixes the background deformation observed with spatial high-resolution tiling.** The recommended path retains the complete token context instead of denoising independent spatial crops. This addresses the background swelling, displacement and instability reproduced in our workflow. It does not promise to eliminate every artifact a video model can generate.
- **Reduces intermediate VRAM demand throughout the target-resolution forward pass.** QKV projections, feed-forward work, model input and final output projections use bounded batches. Sparse attention runs in groups of eight heads while retaining the complete token/key axis.
- **Keeps native attention acceleration.** Global stream works with ComfyUI's native H3 SLA and sol-attn paths, including their conditioning sinks and native dense fallback outside the sparse interval.
- **Avoids the “fast first step, very slow later steps” failure mode we reproduced.** Bounded input/output intermediates and guarded idle-pool trimming reduce competition between temporary allocations and DynamicVRAM's weight cache. Block-local prepared weights also reduce repeated LoRA preparation.
- **Accelerates the learned latent upscaler on affected CUDA/cuDNN configurations.** A guarded Conv3d workaround follows the approach already used by ComfyUI core; other configurations retain the standard fallback.
- **Simplifies the interface.** Main controls stay visible, advanced controls are collapsed by default, inactive controls are hidden, and existing workflow parameter values are preserved.

## Real generation on 16 GB VRAM

Recorded on an **RTX 5070 Ti 16 GB with 128 GB system RAM**, using a native INT8 H3 model, ten Euler steps, six low-resolution steps, scale 0.60, cfg 1, learned latent upscaling, SLA 10%, and Global stream with chunk 4096 / weight cache 4 GiB.

| Requested video | Target resolution | Complete workflow time | Result |
| --- | --- | --- | --- |
| 15 seconds | 1376 × 768, 1.0 MP selector | 5m 27s – 5m 32s, three runs | Success |
| **20 seconds** | **1376 × 768, 1.0 MP selector** | **7m 49s / 8m 12s, two runs** | **Success** |
| **15 seconds** | **1920 × 1088, 2.0 MP selector** | **14m 08s** | **Completed in the execution log** |

These are complete workflow execution times from ComfyUI history and logs, not single-layer estimates. They are individual completed runs with different content, not a controlled speed comparison.

### 2.0 MP: higher-resolution generation and recorded improvement

The same 16 GB GPU also completed **15 seconds at 1920 × 1088**, with **219709 packed tokens**. The optimized upscaler's complete transition lift took **10.408 s**; the four target-resolution steps took **157.913 / 101.975 / 112.816 / 100.402 s**, and the complete workflow took **14m 08s**.

| 1920 × 1088 run | Whole workflow | Target-resolution stage |
| --- | ---: | ---: |
| Early global build, 10 seconds | 27m 50s | 1322.042 s |
| Optimized stream, before the Conv3d upscaler fix, 10 seconds | **12m 55s** | **249.757 s** |
| Stream + upscaler fix, longer 15-second video | **14m 08s** | 477.195 s |

For the two 10-second records, observed whole-workflow time fell by about **54%** (about **2.15×** the recorded throughput), and the target-resolution stage by about **81%**. These compare development runs of this project, not facok's unmodified sampler; prompts, references, cache state and runtime conditions were not held fixed. The 15-second result demonstrates the larger workload and is not used to calculate those percentages. See [the 2.0 MP measurement details](docs/BENCHMARKS.md#a2-20-mp-high-resolution-results).

The recorded long-video results use substantial system RAM and DynamicVRAM offloading. “16 GB” describes the tested GPU capacity, not a claim that the whole model and workflow fit in 16 GB without offloading. Other GPUs, RAM capacities, models, references, attention settings and video contents can change the result.

## Measured upscaler improvement

Same real upscaler checkpoint, input shape, output shape and temporal windows; isolated GPU test:

| Measurement | Previous Conv3d path | Guarded optimized path |
| --- | ---: | ---: |
| Upscaler network time | 74.834 s | **7.474 s** |
| PyTorch peak allocated | 10.005 GiB | **2.539 GiB** |
| PyTorch peak reserved | 21.537 GiB | **4.889 GiB** |

That is about **10× faster for this upscaler stage**, with approximately **75% lower peak allocated memory** and **77% lower peak reserved memory** in this test. These are PyTorch allocator counters for the isolated test, not whole-video GPU peaks or a 10× end-to-end speed claim. See [benchmark conditions and numerical differences](docs/BENCHMARKS.md).

## Installation

Requires a current ComfyUI build with native MiniMax H3 support. The recommended mode also requires its native sparse-attention and Comfy Kitchen support. The code uses current H3 internals; it is not a generic replacement for every ComfyUI or H3 wrapper version.

1. If the original SelfLift is installed, back it up and move that backup **outside `custom_nodes`**. Keep only one active copy: both projects deliberately register the same backend node IDs for workflow compatibility.
2. Clone this repository into `ComfyUI/custom_nodes`:

   ```bash
   git clone https://github.com/Fish4AI/comfyui-SelfLift-Fish.git
   ```

   The repository is public. No GitHub login is required to clone it over HTTPS.
3. Install the optional [H3 latent upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) checkpoint under `ComfyUI/models/latent_upscale_models/`. The tested selection is `h3_upscaler_lms_v0.1.safetensors`. Weights are not bundled.
4. Restart ComfyUI and hard-refresh the browser. Search for **selflift-Fish** in the `selflift` category.

No additional model weights or training are required by the memory-saving implementation itself. Model, VAE, attention dependencies and any selected external upscaler remain required.

## Recommended starting configuration

| Control | Starting value |
| --- | --- |
| Upstream total steps / sampler | 10 / standard Euler, `s_churn=0` |
| Low-resolution steps | 6, leaving 4 target-resolution steps |
| Low-resolution scale | 0.60 |
| cfg | 1.0 |
| Upscaler model | `h3_upscaler_lms_v0.1.safetensors` |
| Pixel correction coverage | 0 |
| Stage 2 memory saving | On |
| Memory-saving mode | Global stream |
| Token chunk size | 4096 |
| Weight cache budget | 4 GiB |
| Unload upscaler | On |

Configure SLA/sol-attn in the official **Model Sparse Attention** node upstream. SLA 10% is the tested baseline; use attention settings appropriate to your model/LoRA. The scheduler remains upstream; the recorded runs used `beta`. Total steps, duration and output dimensions are not duplicated in this node.

## Choosing a memory-saving mode

| Mode | Purpose |
| --- | --- |
| **Global stream** | Recommended for native H3 SLA/sol-attn. Full token context with grouped GPU attention and bounded input/output/projection batches. |
| Global | Compatibility path, including native models without a supported sparse patch. Retains global context; memory use can be higher. |
| Spatial / RAM staging | Accepted for old workflows only. Spatial crops can reproduce the background issue; RAM staging was slower in our measured workload. Not offered as new selections. |

Memory saving can help long videos at ordinary resolutions too. Whether it speeds up a task depends on memory pressure and execution overhead; it is not a universal speed switch.

## Compatibility and limits

- Backend ID `SelfLiftH3Sampler`, input names and serialized widget order are retained. The frontend changes labels and layout rather than rewriting saved parameter values. Advanced settings collapse on load; their values remain active.
- Global-context processing preserves access to the complete scene. SLA/sol-attn are still approximate sparse attention; this is not a claim of mathematical equivalence to dense attention or bitwise-identical output.
- Current Global stream accepts the implemented standard floating-point and native TensorWise INT8 QKV paths. Arbitrary quantization formats, custom QKV transforms and other attention overrides are not supported.
- **VSA is not adapted to this project's global paths.** TST/other attention overrides and `noise_mask` also have compatibility restrictions. See [settings and troubleshooting](docs/SETTINGS.md).
- Only standard Euler is supported. No ancestral Euler, DPM++ or other sampler substitution.
- The H3 adaptation is experimental; the original SelfLift paper evaluates image backbones, not this audio-video implementation.
- The latest compact UI was checked against the locally installed ComfyUI frontend. Older frontends may lack the layout APIs it uses.

## Credits and provenance

The progressive-resolution sampler, SelfLift-zero logic, original H3 adaptation, spatial tiling, TST port and latent-upscaler integration originate from **[facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift)**. The inspected local base commit is `835c3cf7919410f71f414688f049c6ca497789d0`.

The Fish changes focus on global-context memory management, native sparse compatibility, input/output batching, weight preparation/cache budgeting, the affected upscaler Conv3d path and frontend usability. This is an independently maintained derivative, not an official facok or MiniMax release.

Thanks also to [LBH-123-AI](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) for the learned H3 upscaler, [ComfyUI](https://github.com/Comfy-Org/ComfyUI) and its native attention/runtime work, and the authors of [SelfLift](https://arxiv.org/abs/2609.02036) and [Temporal State Transport](https://arxiv.org/abs/2609.08505).

Original upstream documentation is retained in [English](docs/UPSTREAM_README.md) and [Chinese](docs/UPSTREAM_README_CN.md). See [NOTICE.md](NOTICE.md) for authorship and licensing provenance.
