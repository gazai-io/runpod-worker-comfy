from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np
from PIL import Image
import torch


NODE_PATH = Path(__file__).resolve().parents[1] / "custom_nodes" / "ComfyUI-FeyNoBg-Matte"
SPEC = importlib.util.spec_from_file_location("feynobg_matte", NODE_PATH / "__init__.py")
PACKAGE = importlib.util.module_from_spec(SPEC)


@contextmanager
def mock_modules(modules):
    # Restore only the test doubles. Torch registers operators during lazy imports.
    original = {name: sys.modules[name] for name in modules if name in sys.modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in modules:
            sys.modules.pop(name, None)
        sys.modules.update(original)


class FeyNoBgMatteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder_paths = ModuleType("folder_paths")
        with mock_modules({"folder_paths": cls.folder_paths, SPEC.name: PACKAGE}):
            SPEC.loader.exec_module(PACKAGE)
        cls.nodes = PACKAGE.nodes

    def setUp(self):
        work = self.enterContext(TemporaryDirectory())
        self.models_dir = Path(work)
        self.model_dir = self.models_dir / "RMBG" / "FeyNoBg"
        self.populate_model_dir(self.model_dir)
        self.folder_paths.models_dir = work
        self.folder_paths.get_folder_paths = Mock(side_effect=KeyError)
        self.volume_dir = self.models_dir / "volume" / "FeyNoBg"
        self.enterContext(patch.object(self.nodes, "VOLUME_MODEL_DIR", self.volume_dir))
        self.device = Mock(return_value=torch.device("cpu"))
        comfy = SimpleNamespace(model_management=SimpleNamespace(get_torch_device=self.device))
        self.enterContext(mock_modules({"comfy": comfy}))
        self.nodes._model_cache.clear()
        self.addCleanup(self.nodes._model_cache.clear)
        self.node = PACKAGE.NODE_CLASS_MAPPINGS["FeyNoBgMatte"]()

    def populate_model_dir(self, model_dir):
        model_dir.mkdir(parents=True)
        (model_dir / self.nodes.WEIGHTS).touch()
        (model_dir / self.nodes.MODEL_CONFIG).write_text('{"embed_dim": 192}', encoding="utf-8")
        (model_dir / self.nodes.PROCESSOR_CONFIG).write_text(
            '{"size": {"height": 1024, "width": 1024}, "resample": 2}', encoding="utf-8"
        )

    def mock_reference_pipeline(self, outputs, dtype=torch.float16):
        self.events = []
        self.crops = []
        self.targets = []
        alpha_values = iter(outputs)
        pixels = torch.ones((1, 3, 1, 1), dtype=torch.float32)
        model_output = object()

        def preprocess(image, *, return_tensors):
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertEqual((type(image), image.mode, return_tensors), (Image.Image, "RGB", "pt"))
            self.events.append("preprocess")
            self.crops.append(np.asarray(image))
            return {"pixel_values": pixels}

        def forward(*, pixel_values):
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertEqual((pixel_values.dtype, pixel_values.device), (dtype, self.device.return_value))
            self.events.append("forward")
            return model_output

        def postprocess(outputs, *, target_sizes):
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertIs(outputs, model_output)
            self.assertEqual(target_sizes, [self.crops[-1].shape[:2]])
            self.events.append("postprocess")
            self.targets.extend(target_sizes)
            return [torch.full(target_sizes[0], next(alpha_values), dtype=torch.float32)]

        model = Mock(side_effect=forward)
        model.eval.return_value = model
        model.to.return_value = model
        processor = Mock(side_effect=preprocess)
        processor.post_process_alpha_matting.side_effect = postprocess
        config = Mock()
        constructor = Mock(return_value=model)
        processor_constructor = Mock(return_value=processor)
        weights = Mock(return_value={"decoder.weight": torch.ones(1)})
        self.enterContext(mock_modules({
            "nobg": SimpleNamespace(BiRefNetImageProcessor=processor_constructor),
            "nobg.birefnet.modeling_birefnet": SimpleNamespace(BiRefNet=constructor, BiRefNetConfig=config),
            "safetensors.torch": SimpleNamespace(load_file=weights),
        }))
        return model, config, constructor, processor_constructor, weights

    def assert_output(self, rgba, mask, rgb, expected_alpha):
        self.assertEqual(rgba.shape, (*rgb.shape[:-1], 4))
        self.assertEqual(mask.shape, rgb.shape[:-1])
        self.assertEqual((rgba.dtype, mask.dtype), (torch.float32, torch.float32))
        torch.testing.assert_close(rgba[..., :3], rgb, rtol=0, atol=0)
        torch.testing.assert_close(rgba[..., 3], mask, rtol=0, atol=0)
        torch.testing.assert_close(mask, expected_alpha, rtol=0, atol=0)

    def assert_halves(self, height):
        self.mock_reference_pipeline([0.125, 0.625])
        rgb_bytes = np.arange(height * 4 * 3, dtype=np.uint8).reshape(height, 4, 3)
        rgb = torch.tensor(rgb_bytes, dtype=torch.float32).unsqueeze(0) / 255
        rgba, mask = self.node.matte(rgb)
        split = height // 2
        expected = torch.full((1, height, 4), 0.625)
        expected[:, :split, :] = 0.125
        self.assert_output(rgba, mask, rgb, expected)
        self.assertEqual(self.targets, [(split, 4), (height - split, 4)])
        self.assertEqual(self.events, ["preprocess", "forward", "postprocess"] * 2)
        np.testing.assert_array_equal(self.crops[0], rgb_bytes[:split])
        np.testing.assert_array_equal(self.crops[1], rgb_bytes[split:])

    def test_halves_even_height(self):
        self.assert_halves(6)

    def test_halves_odd_height(self):
        self.assert_halves(5)

    def test_halves_requires_two_rows_before_loading(self):
        with self.assertRaisesRegex(ValueError, "at least two image rows"):
            self.node.matte(torch.zeros((1, 1, 6, 3)))
        self.device.assert_not_called()

    def test_full_batch_rgba_and_strict_load(self):
        model, config, constructor, processor, weights = self.mock_reference_pipeline(
            [0.12345, 0.875], dtype=torch.bfloat16
        )
        rgb_bytes = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
        original = torch.tensor(rgb_bytes, dtype=torch.float32) / 255
        image = torch.cat((original, torch.full((2, 3, 4, 1), 0.999)), dim=-1)
        rgba, mask = self.node.matte(image, mode="full", precision="bf16", invert_mask=True)
        alpha = torch.tensor([0.12345, 0.875]).view(2, 1, 1).expand(2, 3, 4)
        self.assert_output(rgba, mask, original, 1 - alpha)
        np.testing.assert_array_equal(self.crops, rgb_bytes)
        self.assertEqual(self.targets, [(3, 4)] * 2)
        self.assertEqual(self.events, ["preprocess", "forward", "postprocess"] * 2)
        config.assert_called_once_with(embed_dim=192)
        constructor.assert_called_once_with(config.return_value)
        weights.assert_called_once_with(str(self.model_dir / self.nodes.WEIGHTS))
        model.load_state_dict.assert_called_once_with(weights.return_value, strict=True)
        model.eval.assert_called_once_with()
        self.assertEqual(model.to.call_args_list, [call(dtype=torch.bfloat16), call(torch.device("cpu"))])
        processor.assert_called_once_with(size={"height": 1024, "width": 1024}, resample=2)

    def test_full_clips_alpha(self):
        self.mock_reference_pipeline([-0.25, 1.25], dtype=torch.float32)
        rgb = torch.zeros((2, 3, 1, 3), dtype=torch.float32)
        rgba, mask = self.node.matte(rgb, mode="full", precision="fp32")
        self.assert_output(rgba, mask, rgb, torch.tensor([0., 1.]).view(2, 1, 1).expand(2, 3, 1))

    def test_tiles_v2_stride_edge_alignment_and_float32_average(self):
        height, width = 1127, 1941
        values = [0.12345, 0.31234, 0.56215, 0.71234, 0.87654, 0.97259]
        self.mock_reference_pipeline(values)
        rgb_bytes = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_bytes[:, :, 0] = np.arange(width) % 256
        rgb_bytes[:, :, 1] = (np.arange(height) % 256)[:, None]
        rgb = torch.tensor(rgb_bytes, dtype=torch.float32).unsqueeze(0) / 255
        rgba, mask = self.node.matte(rgb, mode="tiles_v2")
        x_starts, y_starts = (0, 896, 917), (0, 103)
        expected_rows = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        for index, (y, x) in enumerate((y, x) for y in y_starts for x in x_starts):
            rows, cols = slice(y, y + 1024), slice(x, x + 1024)
            np.testing.assert_array_equal(self.crops[index], rgb_bytes[rows, cols])
            expected_rows[rows, cols] += values[index]
            counts[rows, cols] += 1
        self.assert_output(rgba, mask, rgb, torch.from_numpy(expected_rows / counts).unsqueeze(0))
        self.assertEqual(self.targets, [(1024, 1024)] * 6)
        self.assertEqual(self.events, ["preprocess", "forward", "postprocess"] * 6)

    def test_tiles_v2_upscales_small_images_and_restores_soft_alpha(self):
        self.mock_reference_pipeline([0.17345, 0.82655])
        rgb_bytes = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        rgb = torch.tensor(rgb_bytes, dtype=torch.float32).unsqueeze(0) / 255
        rgba, mask = self.node.matte(rgb, mode="tiles_v2")
        work = np.asarray(Image.fromarray(rgb_bytes).resize((1280, 1024), Image.Resampling.BILINEAR))
        self.assertEqual(self.targets, [(1024, 1024)] * 2)
        np.testing.assert_array_equal(self.crops[0], work[:, :1024])
        np.testing.assert_array_equal(self.crops[1], work[:, 256:])
        scaled_alpha = np.repeat(np.array([[0.17345] * 256 + [0.5] * 768 + [0.82655] * 256],
                                          dtype=np.float32), 1024, axis=0)
        expected = np.asarray(Image.fromarray(scaled_alpha).resize((5, 4), Image.Resampling.BILINEAR)).copy()
        self.assert_output(rgba, mask, rgb, torch.from_numpy(expected).unsqueeze(0))

    def test_keep_model_loaded_keys_and_default_cuda_eviction(self):
        cuda = torch.device("cuda:0")
        self.device.return_value = cuda
        predict = Mock(return_value=np.zeros((3, 2), dtype=np.float32))
        loader = self.enterContext(patch.object(self.nodes, "_FeyNoBg", return_value=predict))
        clear_cuda = self.enterContext(patch.object(torch.cuda, "empty_cache"))
        image = torch.zeros((1, 3, 2, 3))
        self.node.matte(image, mode="full", keep_model_loaded=True)
        self.node.matte(image, mode="full", keep_model_loaded=True)
        loader.assert_called_once_with(self.model_dir, "fp16", cuda)
        clear_cuda.assert_not_called()

        self.node.matte(image, mode="full", precision="fp32", keep_model_loaded=True)
        self.device.return_value = torch.device("cuda:1")
        self.node.matte(image, mode="full", keep_model_loaded=True)
        self.device.return_value = cuda
        alternate = self.models_dir / "alternate"
        self.populate_model_dir(alternate)
        with patch.object(self.nodes, "_resolve_model_dir", return_value=alternate):
            self.node.matte(image, mode="full", keep_model_loaded=True)
        self.assertEqual(loader.call_args_list, [
            call(self.model_dir, "fp16", cuda),
            call(self.model_dir, "fp32", cuda),
            call(self.model_dir, "fp16", torch.device("cuda:1")),
            call(alternate, "fp16", cuda),
        ])

        self.node.matte(image, mode="full")
        self.assertNotIn((self.model_dir, "fp16", cuda), self.nodes._model_cache)
        clear_cuda.assert_called_once_with()
        self.node.matte(image, mode="full")
        self.assertEqual(loader.call_count, 5)
        self.assertEqual(clear_cuda.call_count, 2)
        self.assertEqual(predict.call_count, 7)

    def test_failed_inference_evicts_model_and_clears_cuda_cache(self):
        self.device.return_value = torch.device("cuda:0")
        loader = self.enterContext(patch.object(self.nodes, "_FeyNoBg"))
        loader.return_value.side_effect = RuntimeError("inference failed")
        with patch.object(torch.cuda, "empty_cache") as clear_cuda:
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                self.node.matte(torch.zeros((1, 2, 2, 3)), mode="full")
        self.assertFalse(self.nodes._model_cache)
        clear_cuda.assert_called_once_with()

    def test_model_resolution_priority_and_complete_directory(self):
        volume = self.volume_dir
        registered = self.models_dir / "shared" / "FeyNoBg"
        lowercase = self.models_dir / "lower" / "FeyNoBg"
        for directory in (volume, registered, lowercase):
            self.populate_model_dir(directory)
        self.folder_paths.get_folder_paths.side_effect = lambda name: {
            "RMBG": [str(registered.parent)],
            "rmbg": [str(lowercase.parent)],
        }[name]

        self.enterContext(patch.dict(os.environ, {"FEYNOBG_ALLOW_HF_DOWNLOAD": "1", "HF_HUB_OFFLINE": "1"}))
        download = Mock()
        with mock_modules({"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
            self.assertEqual(self.nodes._resolve_model_dir(), volume)
            (volume / self.nodes.MODEL_CONFIG).unlink()
            self.assertEqual(self.nodes._resolve_model_dir(), self.model_dir)
            (self.model_dir / self.nodes.WEIGHTS).unlink()
            self.assertEqual(self.nodes._resolve_model_dir(), registered)
            (registered / self.nodes.PROCESSOR_CONFIG).unlink()
            self.assertEqual(self.nodes._resolve_model_dir(), lowercase)
        download.assert_not_called()

    def test_model_download_fallback_uses_pinned_snapshot(self):
        (self.model_dir / self.nodes.WEIGHTS).unlink()
        snapshot = self.models_dir / "hub" / "snapshots" / self.nodes.MODEL_REVISION
        self.populate_model_dir(snapshot)
        download = Mock(side_effect=lambda *, filename, **kwargs: str(snapshot / filename))
        self.enterContext(patch.dict(os.environ, {"FEYNOBG_ALLOW_HF_DOWNLOAD": "1"}))
        with mock_modules({"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
            self.assertEqual(self.nodes._resolve_model_dir(), snapshot)
        self.assertEqual(download.call_args_list, [
            call(repo_id="1038lab/FeyNobg", filename=filename, revision="59dd27c65909b7dcc2de8c3239712b0b997c3e41")
            for filename in ("FeyNobg.safetensors", "config.json", "preprocessor_config.json")
        ])

    def test_missing_model_lists_local_paths_without_download(self):
        (self.model_dir / self.nodes.WEIGHTS).unlink()
        shared = self.models_dir / "shared"
        self.folder_paths.get_folder_paths.side_effect = None
        self.folder_paths.get_folder_paths.return_value = [str(shared)]
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("FEYNOBG_ALLOW_HF_DOWNLOAD", None)
        download = Mock()
        with mock_modules({"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
            with self.assertRaises(FileNotFoundError) as failure:
                self.nodes._resolve_model_dir()
        for path in (self.volume_dir, self.model_dir, shared / "FeyNoBg"):
            self.assertIn(str(path), str(failure.exception))
        for model_file in self.nodes.MODEL_FILES:
            self.assertIn(model_file, str(failure.exception))
        download.assert_not_called()

    @unittest.skipUnless(os.getenv("FEYNOBG_TEST_MODEL_DIR"), "requires a local FeyNobg checkpoint")
    def test_real_checkpoint_matches_model_key_schema(self):
        from nobg.birefnet.modeling_birefnet import BiRefNet, BiRefNetConfig
        from safetensors import safe_open

        directory = Path(os.environ["FEYNOBG_TEST_MODEL_DIR"])
        config = BiRefNetConfig(**json.loads((directory / self.nodes.MODEL_CONFIG).read_text()))
        with torch.device("meta"):
            model = BiRefNet(config)
        target = model.state_dict()
        with safe_open(directory / self.nodes.WEIGHTS, framework="pt") as checkpoint:
            # Read shapes from the real checkpoint header without loading the weights.
            shapes = {key: tuple(checkpoint.get_slice(key).get_shape()) for key in checkpoint.keys()}
        self.assertIn("bb.swin.encoder.layers.0.blocks.0.attention.q_proj.weight", shapes)
        converted = self.nodes._state_dict_for_model(shapes, target)
        self.assertEqual(set(target) - set(converted), set(), "missing model keys")
        self.assertEqual(set(converted) - set(target), set(), "unexpected checkpoint keys")
        for key, value in converted.items():
            shape = value.shape if isinstance(value, torch.Tensor) else value
            self.assertEqual(tuple(target[key].shape), shape, key)


if __name__ == "__main__":
    unittest.main()
