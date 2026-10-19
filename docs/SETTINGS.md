# Settings and troubleshooting

## Main controls

| Display label | Backend key | Effect |
| --- | --- | --- |
| Seed / Seed behavior | `seed` / frontend seed control | Use Fixed and an unchanged seed for comparisons. |
| Low-resolution steps | `transition_step` | Number of early low-resolution evaluations, not total steps. Six of ten leaves four target-resolution steps. Increasing it usually reduces work but leaves less target-resolution refinement. Must be less than upstream total steps. |
| Low-resolution scale | `lowres_scale` | Width and height fraction. 0.60 means about 36% of video spatial positions after rounding; duration stays unchanged. Smaller can lose small details and is not a guaranteed proportional compute reduction. |
| Upscaler model | `upscaler_model` | Learned latent lift at the transition. Changing weights changes reconstruction and performance. None requires a usable pixel-correction route. |
| Stage 2 memory saving | `highres_tiling` | Enables the selected target-resolution memory path. Mode/chunk/cache are inactive when off. |

The total sigma schedule, duration, output dimensions and sparse-attention mode
remain upstream controls. No duplicate total-step or resolution controls are
added here.

## Advanced controls

| Display label | Default | Meaning |
| --- | --- | --- |
| cfg | 1.0 | Conditioning guidance. Moving away from 1 can enable negative-conditioning evaluations and increase work; higher is not a quality level. |
| Pixel correction coverage | 0 | `rho`: fraction of high-risk latent locations corrected toward a pixel-VAE anchor. Not blend strength. Positive coverage and positive maximum strength invoke video decode, resize and re-encode, even for small coverage. |
| Correction strength min / max | 0.5 / 1.0 | `w_min/w_max`: blend strength within selected locations. Min must not exceed max. Inactive at coverage 0; max 0 also disables the pixel branch. |
| Unload upscaler | On | Releases upscaler residency after the lift. Recommended for the tested 16 GB system. |
| Memory-saving mode | Global stream | Recommended native sparse path; Global is the broader compatibility alternative. |
| Token chunk size | 4096 | Projection/MLP batches, also input/output batches in Global stream. Smaller generally reduces intermediate demand but adds overhead. It is not an attention window, spatial crop or frame count. It does not set the upscaler's temporal window. |
| Weight cache budget (GiB) | 4 | DynamicVRAM model-weight residency budget for Global stream. Not a total-VRAM or RAM limit. Too little can increase transfers; too much can crowd workspace. 0 retains automatic policy, not zero caching. |

Advanced settings default to collapsed, including after loading workflows.
Collapsing does not reset values. The button reports non-default cfg, active
pixel correction or required correction when no learned upscaler is selected.
Selecting None expands the section without silently changing settings.

Inactive mode/chunk/cache controls are hidden when memory saving is off; cache
is hidden outside Global stream; correction strengths are hidden at coverage 0.
Hidden values stay serialized. Converted widget inputs remain controlled by
ComfyUI so existing connections are preserved.

## Inputs and supported paths

- `model_hires` optionally uses a second model/LoRA stack at target resolution.
  It must share the architecture and latent format. Leave unconnected to reuse
  the main model.
- `vae` is needed by the pixel-anchor route. Keep the connector for compatibility
  even when coverage 0 skips that route.
- Global stream requires ComfyUI native H3 sparse patches (SLA or sol-attn).
  VSA is explicitly not adapted. Global can use the native dense path.
- Global-context chunking does not support arbitrary attention overrides,
  including TST, and the sampler rejects `noise_mask` with memory saving on.
- Native TensorWise INT8 and the implemented standard floating-point QKV paths
  are supported. Other quantization layouts/custom transforms may be rejected.
- The image sampler and TST node from upstream are retained. Their presence does
  not mean all combinations with the H3 global paths are supported.

## Common issues

**Duplicate nodes or wrong frontend:** keep one SelfLift installation, move
backups outside `custom_nodes`, restart ComfyUI and hard-refresh the browser.
The backend ID remains `SelfLiftH3Sampler`.

**Global stream reports a patch/quantization mismatch:** use a supported native
H3/SLA/sol-attn setup, or use Global where applicable. Do not silently replace
the sparse mode or model to obtain a superficially successful run.

**Long videos still run slowly or exhaust memory:** keep cfg at the model's
appropriate value, examine duration/resolution/reference tokens, inspect the
timing logs, and tune chunk/cache within available VRAM. Host RAM and weight
offloading remain relevant. Shared GPU memory counters include mapped/staged
memory and do not by themselves prove driver paging.

**Changing pixel coverage unexpectedly slows the transition:** any enabled
pixel-anchor route needs VAE decode/re-encode. Coverage 0 with a learned
upscaler is the tested default. None plus coverage 0 disables both useful routes
and is rejected.

**Background instability:** verify memory saving uses Global stream or Global,
not a restored legacy Spatial value. Keep seed, prompt and attention settings
fixed when comparing. Global context addresses the independent-crop issue;
model-generated artifacts can still occur.

**Profiling:** `[SelfLift timing]` reports complete step/stage wall time.
`[SelfLift global stream forward]` is a forward diagnostic, not the complete
workflow timer. Optional `SELFLIFT_TIMING_SYNC`, `SELFLIFT_MEMORY_LOG`,
`SELFLIFT_DEBUG` and `SELFLIFT_TST_EXACT` are diagnostic features; extra decoding
or synchronization can add time and memory use.
