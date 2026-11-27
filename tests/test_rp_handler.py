import unittest
from unittest.mock import patch, MagicMock, mock_open, Mock
import sys
import os
import json
import base64
import tempfile

# Make sure that "src" is known and can be used to import rp_handler.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from src import rp_handler

# Local folder for test resources
RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES = "./test_resources/images"


class TestRunpodWorkerComfy(unittest.TestCase):
    def test_valid_input_with_workflow_only(self):
        input_data = {"workflow": {"key": "value"}}
        validated_data, error = rp_handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(validated_data, {"download_file_names": [], "workflow": {"key": "value"}, "images": None})

    def test_valid_input_with_workflow_and_images(self):
        input_data = {
            "download_file_names": [],
            "workflow": {"key": "value"},
            "images": [{"name": "image1.png", "image": "base64string"}],
        }
        validated_data, error = rp_handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(validated_data, input_data)

    def test_input_missing_workflow(self):
        input_data = {"images": [{"name": "image1.png", "image": "base64string"}]}
        validated_data, error = rp_handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Missing 'workflow' parameter")

    def test_input_with_invalid_images_structure(self):
        input_data = {
            "workflow": {"key": "value"},
            "images": [{"name": "image1.png"}],  # Missing 'image' key
        }
        validated_data, error = rp_handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(
            error, "'images' must be a list of objects with 'name' and 'image' keys"
        )

    def test_invalid_json_string_input(self):
        input_data = "invalid json"
        validated_data, error = rp_handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Invalid JSON format in input")

    def test_valid_json_string_input(self):
        input_data = '{"workflow": {"key": "value"}}'
        validated_data, error = rp_handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(validated_data, {"download_file_names": [], "workflow": {"key": "value"}, "images": None})

    def test_empty_input(self):
        input_data = None
        validated_data, error = rp_handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Please provide input")

    @patch("rp_handler.requests.get")
    def test_check_server_server_up(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.return_value = mock_response

        result = rp_handler.check_server("http://127.0.0.1:8188", 1, 50)
        self.assertTrue(result)

    @patch("rp_handler.requests.get")
    def test_check_server_server_down(self, mock_requests):
        mock_requests.get.side_effect = rp_handler.requests.RequestException()
        result = rp_handler.check_server("http://127.0.0.1:8188", 1, 50)
        self.assertFalse(result)

    @patch("rp_handler.urllib.request.urlopen")
    def test_queue_prompt(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"prompt_id": "123"}).encode()
        mock_urlopen.return_value = mock_response
        result = rp_handler.queue_workflow({"prompt": "test"})
        self.assertEqual(result, {"prompt_id": "123"})

    @patch("rp_handler.urllib.request.urlopen")
    def test_get_history(self, mock_urlopen):
        # Mock response data as a JSON string
        mock_response_data = json.dumps({"key": "value"}).encode("utf-8")

        # Define a mock response function for `read`
        def mock_read():
            return mock_response_data

        # Create a mock response object
        mock_response = Mock()
        mock_response.read = mock_read

        # Mock the __enter__ and __exit__ methods to support the context manager
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = Mock()

        # Set the return value of the urlopen mock
        mock_urlopen.return_value = mock_response

        # Call the function under test
        result = rp_handler.get_history("123")

        # Assertions
        self.assertEqual(result, {"key": "value"})
        mock_urlopen.assert_called_with("http://127.0.0.1:8188/history/123")

    @patch("builtins.open", new_callable=mock_open, read_data=b"test")
    def test_base64_encode(self, mock_file):
        test_data = base64.b64encode(b"test").decode("utf-8")

        result = rp_handler.base64_encode("dummy_path")

        self.assertEqual(result, test_data)

    @patch("rp_handler.os.path.exists")
    @patch("rp_handler.rp_upload.upload_image")
    @patch.dict(
        os.environ, {"COMFY_OUTPUT_PATH": RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES}
    )
    def test_bucket_endpoint_not_configured(self, mock_upload_image, mock_exists):
        mock_exists.return_value = True
        mock_upload_image.return_value = "simulated_uploaded/image.png"

        outputs = {
            "node_id": {"images": [{"filename": "ComfyUI_00001_.png", "subfolder": ""}]}
        }
        job_id = "123"

        result = rp_handler.process_output_images(outputs, job_id, [])

        self.assertEqual(result["status"], "success")

    @patch("rp_handler.os.path.exists")
    @patch("src.rp_handler.runpod_upload_image")
    @patch.dict(
        os.environ,
        {
            "COMFY_OUTPUT_PATH": RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES,
            "BUCKET_ENDPOINT_URL": "http://example.com",
        },
    )
    def test_bucket_endpoint_configured(self, mock_upload_image, mock_exists):
        # Mock the os.path.exists to return True, simulating that the image exists
        mock_exists.return_value = True

        # Current handler uses runpod_upload_image, which returns (url, obj_key)
        mock_upload_image.return_value = (
            "http://example.com/uploaded/image.png",
            "runpod-temp/123/image.png",
        )

        # Define the outputs and job_id for the test
        outputs = {"node_id": {"images": [{"filename": "ComfyUI_00001_.png", "subfolder": "test"}]}}
        job_id = "123"

        # Call the function under test
        result = rp_handler.process_output_images(outputs, job_id, [])

        # Assertions
        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["message"],
            [{"url": "http://example.com/uploaded/image.png", "obj_key": "runpod-temp/123/image.png"}],
        )
        self.assertEqual(result["texts"], [])
        self.assertEqual(result["text_outputs"], {})
        mock_upload_image.assert_called_once_with(
            job_id,
            "./test_resources/images/test/ComfyUI_00001_.png",
            bucket_name="runpod-temp",
        )

    @patch("rp_handler.os.path.exists")
    @patch("src.rp_handler.runpod_upload_image")
    @patch.dict(
        os.environ,
        {
            "COMFY_OUTPUT_PATH": RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES,
            "BUCKET_ENDPOINT_URL": "http://example.com",
            "BUCKET_ACCESS_KEY_ID": "",
            "BUCKET_SECRET_ACCESS_KEY": "",
        },
    )
    def test_bucket_image_upload_fails_env_vars_wrong_or_missing(
        self, mock_upload_image, mock_exists
    ):
        # Simulate the file existing in the output path
        mock_exists.return_value = True

        # When AWS credentials are wrong or missing, runpod_upload_image
        # returns a (url, obj_key) pair under simulated_uploaded/.
        mock_upload_image.return_value = (
            "simulated_uploaded/image.png",
            "simulated_uploaded/image.png",
        )

        outputs = {
            "node_id": {"images": [{"filename": "ComfyUI_00001_.png", "subfolder": ""}]}
        }
        job_id = "123"

        result = rp_handler.process_output_images(outputs, job_id, [])

        # Check if the image was saved to the 'simulated_uploaded' directory
        self.assertIn("simulated_uploaded", result["message"][0]["url"])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["texts"], [])
        self.assertEqual(result["text_outputs"], {})

    @patch("rp_handler.requests.post")
    def test_upload_images_successful(self, mock_post):
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.text = "Successfully uploaded"
        mock_post.return_value = mock_response

        test_image_data = base64.b64encode(b"Test Image Data").decode("utf-8")

        images = [{"name": "test_image.png", "image": test_image_data}]

        responses = rp_handler.upload_images(images)

        self.assertEqual(len(responses), 3)
        self.assertEqual(responses["status"], "success")

    @patch("rp_handler.requests.post")
    def test_upload_images_failed(self, mock_post):
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 400
        mock_response.text = "Error uploading"
        mock_post.return_value = mock_response

        test_image_data = base64.b64encode(b"Test Image Data").decode("utf-8")

        images = [{"name": "test_image.png", "image": test_image_data}]

        responses = rp_handler.upload_images(images)

        self.assertEqual(len(responses), 3)
        self.assertEqual(responses["status"], "error")


    def test_image_only_outputs_have_empty_texts(self):
        outputs = {
            "6": {"images": [{"filename": "ComfyUI_00001_.png", "subfolder": ""}]}
        }
        workflow = {"6": {"class_type": "SaveImage", "inputs": {}}}
        with patch.dict(
            os.environ,
            {"COMFY_OUTPUT_PATH": RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES},
        ):
            result = rp_handler.process_output_images(outputs, "123", [], workflow=workflow)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["success"])
        self.assertEqual(len(result["message"]), 1)
        self.assertEqual(result["download_files"], [])
        self.assertEqual(result["texts"], [])
        self.assertEqual(result["text_outputs"], {})

    def test_preview_any_and_show_text_are_ignored(self):
        outputs = {
            "4": {"text": ["1girl, blonde hair, amber eyes"]},
            "5": {"text": ["debug: intermediate prompt"]},
        }
        workflow = {
            "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}},
            "5": {"class_type": "ShowText|pysssss", "inputs": {"text": ["3", 0]}},
        }
        result = rp_handler.process_output_images(outputs, "123", [], workflow=workflow)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], [])
        self.assertEqual(result["download_files"], [])
        self.assertEqual(result["texts"], [])
        self.assertEqual(result["text_outputs"], {})

    def test_inline_text_without_workflow_is_ignored(self):
        # No workflow means class_type cannot be determined, so preview-shaped
        # {"text": [...]} history must not leak into the result.
        outputs = {"4": {"text": ["1girl, blonde hair, ..."]}}
        result = rp_handler.process_output_images(outputs, "123", [])

        self.assertTrue(result["success"])
        self.assertEqual(result["message"], [])
        self.assertEqual(result["texts"], [])
        self.assertEqual(result["text_outputs"], {})

    def test_core_save_text_file_is_collected_and_not_in_message(self):
        tags = "1girl, blonde hair, amber eyes"
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "tags_00001.txt"), "w", encoding="utf-8") as handle:
                handle.write(tags)
            outputs = {
                "8": {
                    "text": [tags],
                    "files": [
                        {"filename": "tags_00001.txt", "subfolder": "", "type": "output"}
                    ],
                }
            }
            workflow = {"8": {"class_type": "SaveText", "inputs": {}}}
            with patch.dict(os.environ, {"COMFY_OUTPUT_PATH": tmp}):
                result = rp_handler.process_output_images(
                    outputs, "123", [], workflow=workflow
                )

        self.assertTrue(result["success"])
        self.assertEqual(result["message"], [])
        self.assertEqual(result["download_files"], [])
        # File contents are preferred over the duplicate ui text, so no duplication.
        self.assertEqual(result["texts"], [tags])
        self.assertEqual(result["text_outputs"], {"8": [tags]})

    def test_was_save_text_file_inline_text_is_collected(self):
        tags = "1girl, brown hair, gray eyes"
        outputs = {"9": {"text": [tags]}}
        workflow = {"9": {"class_type": "Save Text File", "inputs": {}}}
        result = rp_handler.process_output_images(outputs, "123", [], workflow=workflow)

        self.assertTrue(result["success"])
        self.assertEqual(result["message"], [])
        self.assertEqual(result["texts"], [tags])
        self.assertEqual(result["text_outputs"], {"9": [tags]})

    def test_multiple_save_text_nodes_and_string_keys(self):
        outputs = {
            "10": {"string": ["tags from SaveText|pysssss"]},
            "11": {"strings": ["first SaveStringKJ line", "second SaveStringKJ line"]},
        }
        workflow = {
            "10": {"class_type": "SaveText|pysssss", "inputs": {}},
            "11": {"class_type": "SaveStringKJ", "inputs": {}},
        }
        result = rp_handler.process_output_images(outputs, "123", [], workflow=workflow)

        self.assertTrue(result["success"])
        self.assertEqual(result["message"], [])
        self.assertEqual(
            result["texts"],
            [
                "tags from SaveText|pysssss",
                "first SaveStringKJ line",
                "second SaveStringKJ line",
            ],
        )
        self.assertEqual(
            result["text_outputs"],
            {
                "10": ["tags from SaveText|pysssss"],
                "11": ["first SaveStringKJ line", "second SaveStringKJ line"],
            },
        )

    def test_mixed_image_and_opted_in_text_outputs(self):
        tags = "1girl, blonde hair"
        with tempfile.TemporaryDirectory() as tmp:
            image_src = os.path.join(
                RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES, "ComfyUI_00001_.png"
            )
            image_dest = os.path.join(tmp, "ComfyUI_00001_.png")
            with open(image_src, "rb") as src, open(image_dest, "wb") as dest:
                dest.write(src.read())
            with open(os.path.join(tmp, "tags_00001.txt"), "w", encoding="utf-8") as handle:
                handle.write(tags)
            outputs = {
                "6": {"images": [{"filename": "ComfyUI_00001_.png", "subfolder": ""}]},
                "8": {
                    "text": [tags],
                    "files": [
                        {"filename": "tags_00001.txt", "subfolder": "", "type": "output"}
                    ],
                },
                "4": {"text": ["preview text that must be ignored"]},
            }
            workflow = {
                "6": {"class_type": "SaveImage", "inputs": {}},
                "8": {"class_type": "SaveText", "inputs": {}},
                "4": {"class_type": "PreviewAny", "inputs": {}},
            }
            with patch.dict(os.environ, {"COMFY_OUTPUT_PATH": tmp}):
                result = rp_handler.process_output_images(
                    outputs, "123", [], workflow=workflow
                )

        self.assertTrue(result["success"])
        self.assertEqual(len(result["message"]), 1)
        self.assertEqual(result["download_files"], [])
        self.assertEqual(result["texts"], [tags])
        self.assertEqual(result["text_outputs"], {"8": [tags]})

    def test_non_string_text_items_and_unrelated_keys_are_ignored(self):
        tags = "1girl, blonde hair"
        outputs = {
            "8": {
                "text": [tags, {"not": "a string"}, 123],
                "ui_meta": ["should be ignored"],
                "string": [None],
            },
            "4": {"ui_meta": ["foo"], "text": ["preview leak"]},
        }
        workflow = {
            "8": {"class_type": "SaveText", "inputs": {}},
            "4": {"class_type": "PreviewAny", "inputs": {}},
        }
        result = rp_handler.process_output_images(outputs, "123", [], workflow=workflow)

        self.assertTrue(result["success"])
        self.assertEqual(result["message"], [])
        self.assertEqual(result["texts"], [tags])
        self.assertEqual(result["text_outputs"], {"8": [tags]})

    def test_temp_text_files_are_ignored_and_not_added_to_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "preview.txt"), "w", encoding="utf-8") as handle:
                handle.write("temp preview text")
            outputs = {
                "4": {
                    "files": [
                        {"filename": "preview.txt", "subfolder": "", "type": "temp"}
                    ],
                    "text": ["temp preview text"],
                }
            }
            workflow = {"4": {"class_type": "PreviewAny", "inputs": {}}}
            with patch.dict(
                os.environ, {"COMFY_OUTPUT_PATH": tmp, "COMFY_TEMP_PATH": tmp}
            ):
                result = rp_handler.process_output_images(
                    outputs, "123", [], workflow=workflow
                )

        self.assertTrue(result["success"])
        self.assertEqual(result["message"], [])
        self.assertEqual(result["texts"], [])
        self.assertEqual(result["text_outputs"], {})

    def test_text_output_file_without_workflow_is_still_collected(self):
        # File items with a text extension and type == "output" are collected
        # even when workflow/class_type is unavailable.
        tags = "1girl, side ponytail"
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "tags.txt"), "w", encoding="utf-8") as handle:
                handle.write(tags)
            outputs = {
                "8": {
                    "files": [
                        {"filename": "tags.txt", "subfolder": "", "type": "output"}
                    ]
                }
            }
            with patch.dict(os.environ, {"COMFY_OUTPUT_PATH": tmp}):
                result = rp_handler.process_output_images(outputs, "123", [])

        self.assertTrue(result["success"])
        self.assertEqual(result["message"], [])
        self.assertEqual(result["texts"], [tags])
        self.assertEqual(result["text_outputs"], {"8": [tags]})
