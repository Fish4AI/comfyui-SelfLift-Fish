# selflift-Fish

**Longer videos · Higher resolution · Stable backgrounds**

A memory-saving MiniMax H3 sampler for ComfyUI, built on [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift).

[中文](README_CN.md) · [Settings](docs/SETTINGS.md) · [Benchmarks](docs/BENCHMARKS.md) · [Credits](NOTICE.md)

## Highlights

- **Long videos on 16 GB VRAM:** tested with 20 seconds at 1376 × 768.
- **Higher resolution:** completed 15 seconds at 1920 × 1088 on the same GPU.
- **Stable backgrounds:** fixes the swelling and displacement reproduced with spatial tiling in our tests.
- **Attention acceleration retained:** supports native H3 SLA and sol-attn.
- **Faster upscaling and reference generation**, without reducing reference resolution or sampling steps.
- **Simple controls:** advanced settings collapse by default; existing SelfLift workflow parameters are retained.

<p align="center"><img src="docs/assets/compact-node.png" alt="selflift-Fish node" width="480"></p>

## Real results

RTX 5070 Ti **16 GB** · **128 GB system RAM** · native INT8 H3 · Euler 10 steps · SLA 10%.

| Video | Resolution | Complete workflow |
| :--- | :--- | ---: |
| 15 seconds | 1376 × 768 | 5m 27s–5m 32s |
| **20 seconds** | **1376 × 768** | **7m 49s / 8m 12s** |
| **15 seconds** | **1920 × 1088** | **14m 08s** |

Completed runs using RAM offloading; results depend on hardware and workflow.

**Recorded improvements**
- **2.0 MP:** a 10-second workflow went from **27m 50s to 12m 55s** across development versions. These were not controlled A/B runs.
- **Latent upscaling:** **74.8 → 7.5 seconds** in an isolated upscaler test.
- **Latest reference-image update:** **124.6 → 119.1 seconds (4.4% less time)** in a two-reference, 5.17-second video comparison; high-resolution sampling was **6.4% faster**. Excludes initial condition encoding and model loading.

The reference update passed a deterministic full-video output comparison. See [test conditions and limits](docs/BENCHMARKS.md#reference-image-update-2026-10-30).

## Install

Requires ComfyUI with native MiniMax H3 support; Global stream also needs native sparse attention and Comfy Kitchen.

1. Back up any existing SelfLift installation **outside `custom_nodes`**. Keep only one active version.
2. Run inside `ComfyUI/custom_nodes`:

   ```bash
   git clone https://github.com/Fish4AI/comfyui-SelfLift-Fish.git
   ```

3. For learned upscaling, download the [H3 latent upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) to `ComfyUI/models/latent_upscale_models/`.
4. Restart ComfyUI, refresh the browser and search for **selflift-Fish**.

## Quick start

| Setting | Starting value |
| :--- | :--- |
| Total steps / sampler / scheduler | 10 / Euler / beta |
| Low-resolution steps / scale | 6 / 0.60 |
| cfg | 1.0 |
| Upscaler | `h3_upscaler_lms_v0.1.safetensors` |
| Stage 2 memory saving / mode | On / Global stream |
| Token chunk / weight cache | 4096 / 4 GiB |

Configure SLA/sol-attn in the upstream **Model Sparse Attention** node. SLA 10% is our tested baseline. Memory saving also helps longer videos at ordinary resolutions; speed gains depend on memory pressure.

**Compatibility:** standard Euler only; standard floating-point, native INT8 and scaled FP8 E4M3 weights supported. VSA is not supported by the global paths. See [Settings](docs/SETTINGS.md) for advanced controls and other patch restrictions.

## Credits

Built on **[facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift)** and its progressive sampler and H3 integration. Thanks to the SelfLift authors, ComfyUI contributors and LBH-123-AI for the H3 upscaler.

Independently maintained; not an official facok or MiniMax release. [Attribution and licensing](NOTICE.md) · [Original documentation](docs/UPSTREAM_README.md).

**Try it with your workflow—feedback and reproducible issue reports are welcome.**
