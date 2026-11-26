import json
import os
from pathlib import Path
import re
from threading import Lock

import folder_paths
import numpy as np
from PIL import Image
import torch


MODEL_REPO = "1038lab/FeyNobg"
MODEL_REVISION = "59dd27c65909b7dcc2de8c3239712b0b997c3e41"
WEIGHTS = "FeyNobg.safetensors"
MODEL_CONFIG = "config.json"
PROCESSOR_CONFIG = "preprocessor_config.json"
MODEL_FILES = (WEIGHTS, MODEL_CONFIG, PROCESSOR_CONFIG)
VOLUME_MODEL_DIR = Path("/runpod-volume/models/RMBG/FeyNoBg")
SWIN_BLOCK = re.compile(r"(bb\.encoder\.layers\.\d+\.blocks\.\d+\.)(.+)")
SWIN_BLOCK_MODULES = {
    "attention.q_proj": "attention.self.query",
    "attention.k_proj": "attention.self.key",
    "attention.v_proj": "attention.self.value",
    "attention.o_proj": "attention.output.dense",
    "attention.relative_position_bias": "attention.self",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
}
TILE = 1024
STRIDE = 896
PRECISIONS = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

_model_cache = {}
_model_lock = Lock()


def _resolve_model_dir():
    candidates = [
        VOLUME_MODEL_DIR,
        Path(folder_paths.models_dir) / "RMBG" / "FeyNoBg",
    ]
    # The worker's extra_model_paths.yaml registers the lowercase variant.
    for registry_name in ("RMBG", "rmbg"):
        try:
            candidates.extend(Path(path) / "FeyNoBg" for path in folder_paths.get_folder_paths(registry_name))
        except KeyError:
            pass

    for model_dir in candidates:
        if all((model_dir / filename).is_file() for filename in MODEL_FILES):
            return model_dir.resolve()

    if os.environ.get("FEYNOBG_ALLOW_HF_DOWNLOAD") != "1":
        searched = "\n".join(f"  - {path}" for path in dict.fromkeys(candidates))
        raise FileNotFoundError(
            f"FeyNoBgMatte requires all of {', '.join(MODEL_FILES)} in one directory. "
            f"Searched local paths:\n{searched}"
        )

    from huggingface_hub import hf_hub_download

    # The pinned revision places all files in the same Hub cache snapshot.
    paths = [
        hf_hub_download(repo_id=MODEL_REPO, filename=filename, revision=MODEL_REVISION)
        for filename in MODEL_FILES
    ]
    return Path(paths[0]).parent


def _state_dict_for_model(checkpoint, target):
    # The pinned FeyNobg checkpoint uses the Swin layout from Transformers 5.9.
    # The worker's 5.5.4 SwinBackbone registers embeddings and encoder directly.
    patch_weight = "embeddings.patch_embeddings.projection.weight"
    if "bb.swin." + patch_weight not in checkpoint or "bb." + patch_weight not in target:
        return checkpoint

    converted = {}
    for key, tensor in checkpoint.items():
        # SwinModel's final norm affects last_hidden_state only. BiRefNet uses
        # the stage feature maps, whose own trained norms are loaded unchanged.
        if key in ("bb.swin.layernorm.weight", "bb.swin.layernorm.bias"):
            continue
        for prefix in ("embeddings.", "encoder."):
            if key.startswith("bb.swin." + prefix):
                key = "bb." + key.removeprefix("bb.swin.")
                break
        match = SWIN_BLOCK.fullmatch(key)
        if match:
            block, suffix = match.groups()
            module, _, parameter = suffix.rpartition(".")
            if module in SWIN_BLOCK_MODULES:
                key = f"{block}{SWIN_BLOCK_MODULES[module]}.{parameter}"
        if key in converted:
            raise ValueError(f"FeyNoBgMatte checkpoint contains multiple tensors for {key}")
        converted[key] = tensor

    # 5.5.4 persists the deterministic index derived from the window geometry.
    # The checkpoint only stores the learned bias table that the index looks up.
    for key, tensor in target.items():
        if key.endswith(".attention.self.relative_position_index"):
            converted.setdefault(key, tensor)
    return converted


class _FeyNoBg:
    def __init__(self, model_dir, precision, device):
        from nobg import BiRefNetImageProcessor
        from nobg.birefnet.modeling_birefnet import BiRefNet, BiRefNetConfig
        from safetensors.torch import load_file

        self.device = device
        self.dtype = PRECISIONS[precision]
        config = BiRefNetConfig(**json.loads((model_dir / MODEL_CONFIG).read_text(encoding="utf-8")))
        model = BiRefNet(config)
        checkpoint = load_file(str(model_dir / WEIGHTS))
        model.load_state_dict(_state_dict_for_model(checkpoint, model.state_dict()), strict=True)
        del checkpoint
        self.model = model.eval().to(dtype=self.dtype).to(self.device)
        processor_config = json.loads((model_dir / PROCESSOR_CONFIG).read_text(encoding="utf-8"))
        self.processor = BiRefNetImageProcessor(**processor_config)

    @torch.inference_mode()
    def __call__(self, rgb):
        pixels = self.processor(Image.fromarray(rgb), return_tensors="pt")["pixel_values"]
        pixels = pixels.to(self.device).to(dtype=self.dtype)
        outputs = self.model(pixel_values=pixels)
        alpha = self.processor.post_process_alpha_matting(outputs, target_sizes=[rgb.shape[:2]])[0]
        return np.clip(alpha.cpu().float().numpy(), 0, 1)


def _axis_starts(length):
    last_start = length - TILE
    starts = list(range(0, last_start + 1, STRIDE))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _tiles_v2(rgb, predict):
    height, width = rgb.shape[:2]
    scale = max(1.0, TILE / min(width, height))
    work_width, work_height = round(width * scale), round(height * scale)
    resized = (work_width, work_height) != (width, height)
    work = rgb
    if resized:
        work = np.asarray(Image.fromarray(rgb).resize((work_width, work_height), Image.Resampling.BILINEAR))

    total = np.zeros((work_height, work_width), dtype=np.float32)
    counts = np.zeros_like(total)
    for y in _axis_starts(work_height):
        for x in _axis_starts(work_width):
            rows = slice(y, y + TILE)
            columns = slice(x, x + TILE)
            total[rows, columns] += predict(work[rows, columns])
            counts[rows, columns] += 1

    alpha = total / counts
    if resized:
        alpha = np.asarray(Image.fromarray(alpha).resize((width, height), Image.Resampling.BILINEAR))
    return np.clip(alpha, 0, 1)


def _matte(rgb, mode, predict):
    if mode == "halves":
        split = rgb.shape[0] // 2
        return np.concatenate((predict(rgb[:split]), predict(rgb[split:])), axis=0)
    if mode == "tiles_v2":
        return _tiles_v2(rgb, predict)
    return predict(rgb)


class FeyNoBgMatte:
    CATEGORY = "image/matting"
    FUNCTION = "matte"
    RETURN_TYPES = ("IMAGE", "MASK")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["halves", "full", "tiles_v2"], {"default": "halves"}),
                "precision": (list(PRECISIONS), {"default": "fp16"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }

    def matte(self, image, mode="halves", precision="fp16", invert_mask=False, keep_model_loaded=False):
        from comfy import model_management

        if mode == "halves" and image.shape[1] < 2:
            raise ValueError("FeyNoBgMatte halves mode requires at least two image rows.")

        rgb = image[..., :3].detach().to(device="cpu", dtype=torch.float32)
        device = torch.device(model_management.get_torch_device())
        # Hold the lock until inference completes so another call cannot evict an in-use model.
        with _model_lock:
            key = (_resolve_model_dir(), precision, device)
            predict = _model_cache.get(key)
            try:
                if predict is None:
                    predict = _FeyNoBg(*key)
                    _model_cache[key] = predict
                mattes = [
                    _matte(np.rint(frame.numpy() * 255).astype(np.uint8), mode, predict)
                    for frame in rgb
                ]
            finally:
                if not keep_model_loaded:
                    _model_cache.pop(key, None)
                    del predict
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        mask = torch.from_numpy(np.stack(mattes))
        if invert_mask:
            mask = 1 - mask
        return torch.cat((rgb, mask.unsqueeze(-1)), dim=-1), mask
