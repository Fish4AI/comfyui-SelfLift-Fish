# Provenance and acknowledgements

This repository is an independently maintained derivative of
[facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift).

Inspected local upstream base: `835c3cf7919410f71f414688f049c6ca497789d0`.

Upstream contributions include the progressive-resolution engine, SelfLift-zero
correction, original H3 audio-video adaptation, spatial tiling, TST port, model
registration and learned latent-upscaler integration. Original source comments
and the upstream English/Chinese README files are retained.

Fish4AI changes include global-context chunking, native sparse attention
compatibility, grouped GPU head streaming, chunked input/output paths,
prepared weight and cache handling, the guarded upscaler Conv3d workaround,
and the compact frontend.

Related work:

- [SelfLift](https://arxiv.org/abs/2609.02036)
- [Temporal State Transport](https://arxiv.org/abs/2609.08505)
- [LBH-123-AI H3 upscaler weights](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
- [LBH-123-AI original upscaler integration](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)

## Licensing status

No explicit standalone license file was present in the inspected upstream base.
This repository does not relabel upstream code as MIT, Apache or another
license, or grant rights over third-party code or model weights. Original
authors retain their rights; model checkpoints and dependencies have their own
terms. Public source availability does not supply a missing upstream license
or imply unrestricted rights to redistribute or relicense third-party code.
