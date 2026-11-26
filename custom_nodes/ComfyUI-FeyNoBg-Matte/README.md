# FeyNoBg Matte

`FeyNoBgMatte` appears under **image/matting** as **FeyNoBg Matte**. It accepts a float32 ComfyUI `IMAGE` batch in `[0, 1]` and uses the first three channels of RGB or RGBA input.

| Input | Behavior |
| --- | --- |
| `mode` | `halves` (default) splits each image into top and bottom at `H // 2`, with an extra bottom row for odd heights. Each half is restored to its original size before concatenation, with zero overlap and no seam blending. This mode requires at least two rows. `full` runs one pass per image. `tiles_v2` uses 1024×1024 tiles with stride 896, covers the final row and column, and averages overlapping alpha values in float32. Images with a shorter side below 1024 are scaled up for tiling and their mattes are scaled back. |
| `precision` | `fp16` (default), `bf16` or `fp32` on ComfyUI's selected device. |
| `invert_mask` | Defaults to false (foreground 1, background 0). True applies `1 - alpha` to both the `MASK` and the RGBA alpha channel. |
| `keep_model_loaded` | Defaults to false, which evicts the model used by the call and clears the CUDA cache on CUDA. True retains it for calls with the same model directory, precision and device. |

The outputs are float32 `IMAGE` [B, H, W, 4] with original RGB plus soft alpha and float32 `MASK` [B, H, W] with that same alpha. The model processor applies bilinear 1024×1024 resizing and ImageNet normalization to each crop, then restores the matte to the crop size. Alpha stays continuous in `[0, 1]` without thresholding or color refinement.

## Model files

Place `FeyNobg.safetensors`, `config.json` and `preprocessor_config.json` together in the first applicable location.

1. `<ComfyUI models_dir>/RMBG/FeyNoBg`
2. `/runpod-volume/models/RMBG/FeyNoBg`
3. `FeyNoBg` under a registered `RMBG` or `rmbg` model path

When no local directory contains all three files, the first inference downloads the pinned revision to the Hugging Face Hub cache. Set `HF_HUB_CACHE` to a writable persistent directory to reuse the download across containers. The Docker build installs the Python dependencies and node code. [Model files at the pinned revision](https://huggingface.co/1038lab/FeyNobg/tree/59dd27c65909b7dcc2de8c3239712b0b997c3e41)
