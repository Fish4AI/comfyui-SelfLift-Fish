# Audio-reference stability — v0.1.1

Reference audio now triggers one joint audio/video evaluation after the resolution
switch. It is automatic: **10 steps with transition 6 becomes 6 + 5 evaluations**.
Image-only references keep **6 + 4**. Reference-video soundtracks count as audio
references when they have been encoded into the conditioning.

The update retains reference inputs, SLA acceleration and global memory-saving
sampling. Outputs are generated directly; no fixed background plates, compositing
or video stabilization are used.

## Observed results

Two seeds of the same 10.125-second, 1152 × 640 talking-head scene, with identical
reference image/audio, model/LoRA stack and settings: native INT8 H3, Euler/simple
10 steps, transition 6, lowres scale 0.7, cfg 1, SLA 10%, extra tokens 256,
Global stream, token chunk 4096, weight cache 4 GiB. RTX 5070 Ti 16 GB / 128 GB RAM.

| Audio-reference run | Wall/logo local deformation P95 | Sampling time |
|---|---:|---:|
| Seed 95 | 3.08 → 1.17 px | 138.02 → 154.23 s |
| Seed 94 | 1.94 → 1.35 px | 136.24 → 154.76 s |

The wall/logo measure tracks texture and removes common translation, rotation
and zoom. It is not an overall quality score or a measure of absolute camera
movement. Small overall movement remains; not every region improved. The owner
also reviewed the generated results and reported improvement. This does not
establish that all scenes or longer videos will have static backgrounds.

### Image references without reference audio

Four full videos in before/after/after/before order, two seeds, the same dimensions
and settings as above, with one image reference and zero audio references:

| Mean of two runs per version | Before | Updated |
|---|---:|---:|
| Sampling | 131.51 s | 131.34 s |
| Sampling, decoding and export | 153.63 s | 153.10 s |
| Actual evaluations | 6 + 4 | 6 + 4 |

Timing was essentially unchanged. Sampled-frame inspection found no obvious loss
of detail or new major background deformation in these runs. Static wall/shelf
local residual changes were under 0.1 px. Hand gestures, mouths and screen content
can differ, and the generated frame/latent hashes are **not identical**.

All timings above exclude shared text/reference encoding and interpreter startup;
they are real complete-video measurements, not single-layer benchmarks. The image
test's shared encoding took 11.44 s. Two seeds are limited evidence, not a universal
quality guarantee or a statistical performance confidence interval.

## Attention correction and compatibility

A separate correction fixes saturated histogram-bin selection in comfy-kitchen
0.2.40. An oversized candidate set could compete for a capped extra-token list.
In the seed-95 audio test, the 256-token budget admitted up to 18,711 candidates
before truncation; the corrected selection never exceeded 256. Replaying an
identical real layer input became bitwise repeatable. Zero augmentation matched
the native path bitwise.

The bundled native core is selected only for:

- **Windows x64**, **sm_120** GPU (tested on RTX 5070 Ti);
- **comfy-kitchen 0.2.40**;
- a driver reporting **CUDA 13.1 or newer** support.

Other operating systems, architectures, kitchen versions, older drivers or an
unloadable binary keep native attention and log a notice. They still receive the
sampler's audio-reference transition improvement, but **do not receive this native
token-budget correction**. These alternative environments were covered by mocked
dispatch/fallback tests, not full-video GPU validation. No automatic compilation
or dependency replacement is performed.

The core is active only inside SelfLift sampling; other threads/callers keep their
original backend. A CUDA execution error is surfaced rather than retried silently.
Source, build notes and the Apache-2.0 component license are in
[`native/token_guard`](../native/token_guard/README.md).

## Updating

Update the existing installation and restart ComfyUI. Node IDs, saved parameter
order, input names and defaults are unchanged; no new setting is required.
Keep only one active SelfLift installation. The previous release remains available
as [v0.1.0](https://github.com/Fish4AI/comfyui-SelfLift-Fish/releases/tag/v0.1.0).
