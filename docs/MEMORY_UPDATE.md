# v0.1.2 — Stage-two memory cleanup

Global stream sampling can stall when temporary memory from the low-resolution stage and video previews leaves too little device headroom. This update retires completed allocation plans at the resolution switch and reclaims substantial idle workspace when device memory is under pressure.

Native SLA/sol-attn acceleration, model weights, reference conditioning, sampling math and the v0.1.1 audio-reference correction are retained. Existing workflow inputs and defaults are unchanged; no new setting is required.

## Complete workflow test — November 4, 2026

RTX 5070 Ti **16 GB**, **128 GB system RAM**, Windows, native INT8 H3 with two LoRAs, reference image and reference audio, seed 95. Requested duration: 10 seconds; output: **1664 × 928, 243 frames at 24 fps (10.125 seconds)**.

Euler/simple, 10 scheduled steps, transition 6, low-resolution scale 0.70, cfg 1.0, rho 0; the audio-reference correction uses 6 + 5 model evaluations. Global stream, token chunk 4096, weight cache 4 GiB, SLA 10%, extra tokens 256; KJ video preview remains enabled. Tested with PyTorch 2.9.1+cu130, cudaMallocAsync, comfy-aimdo 0.6.0 and comfy-kitchen 0.2.40.

| Measurement | Earlier slow run | Updated installation |
| :--- | ---: | ---: |
| High-resolution step 1 | 1675.694 s | **49.150 s** |
| Step 2 | 50.805 s | 39.905 s |
| Step 3 | 53.578 s | 40.232 s |
| Step 4 | 52.451 s | 40.020 s |
| Step 5 | 428.219 s | **40.060 s** |
| Complete workflow, including encoding and export | 2471.498 s (41m 11s) | **410.402 s (6m 50s)** |

The updated run was executed through the installed ComfyUI server, with no execution-cache hits, and completed successfully. The original settings and seed were preserved; only the output filename prefix changed.

**Output verification:** all **243 decoded video frames and decoded audio were bitwise identical** between these two runs. An independent comparison also matched the complete video/audio latent tensors. All 26 targeted regression tests passed, covering workflow compatibility, audio-reference handling, sparse-token selection and memory-cleanup boundaries.

## What the result means

This addresses memory-pressure stalls, not a fixed speedup for every workflow. The two complete-workflow times are observed runs, not repeated, controlled throughput averages. The original driver's full residency state could not be reconstructed exactly. Stale allocation-plan retention and later-step pressure slowdowns were reproduced separately; identical output was verified for this fixed-seed case, not every GPU or workflow.

In the installed test, retiring the completed low-stage plan released about **6.01 GiB** of graph-owned workspace in **0.278 s**. Subsequent preview-related allocator pools of roughly 4–5 GiB were trimmed to about 0.81–0.84 GiB. These are workspace observations, not whole-run VRAM peaks. Shared GPU memory alone does not establish driver paging.

The allocation-plan reset uses ComfyUI's native cleanup interface and skips active, small or unavailable graphs. Other stacks may see different benefits. The existing [optional native attention correction's hardware requirements](AUDIO_REFERENCE_UPDATE.md) remain unchanged.

Update your existing installation and **restart ComfyUI**. Keep one active SelfLift installation. Feedback with workflow settings and timing logs is welcome.
