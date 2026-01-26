import random
import threading
import traceback
import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.request
import boto3
import time
import os
import requests
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple
import uuid

import websocket


# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = int(os.environ.get("COMFY_API_AVAILABLE_INTERVAL_MS", 50))
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = int(os.environ.get("COMFY_API_AVAILABLE_MAX_RETRIES", 500))
# Time to wait between poll attempts in milliseconds
COMFY_POLLING_TIMEOUT_MS = int(os.environ.get("COMFY_POLLING_TIMEOUT_MS", 1200000))
COMFY_POLLING_INTERVAL_MS = int(os.environ.get("COMFY_POLLING_INTERVAL_MS", 100))
# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
# Enforce a clean state after each job is done
# see https://docs.runpod.io/docs/handler-additional-controls#refresh-worker
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"


def validate_input(job_input):
    """
    Validates the input for the handler function.

    Args:
        job_input (dict): The input data to validate.

    Returns:
        tuple: A tuple containing the validated data and an error message, if any.
               The structure is (validated_data, error_message).
    """
    # Validate if job_input is provided
    if job_input is None:
        return None, "Please provide input"

    # Check if input is a string and try to parse it as JSON
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # Validate 'workflow' in input
    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    # Validate 'images' in input, if provided
    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            "name" in image and "image" in image for image in images
        ):
            return (
                None,
                "'images' must be a list of objects with 'name' and 'image' keys",
            )

    # Return validated data and no error
    return {"workflow": workflow, "images": images, "download_file_names": job_input.get("download_file_names", [])}, None


def check_server(url, retries=500, delay=50):
    """
    Check if a server is reachable via HTTP GET request

    Args:
    - url (str): The URL to check
    - retries (int, optional): The number of times to attempt connecting to the server. Default is 50
    - delay (int, optional): The time in milliseconds to wait between retries. Default is 500

    Returns:
    bool: True if the server is reachable within the given number of retries, otherwise False
    """

    for i in range(retries):
        try:
            response = requests.get(url)

            # If the response status code is 200, the server is up and running
            if response.status_code == 200:
                print(f"runpod-worker-comfy - API is reachable")
                return True
        except requests.RequestException as e:
            # If an exception occurs, the server may not be ready
            print(f"runpod-worker-comfy - API not reachable yet (attempt {i + 1}/{retries})")
            pass

        # Wait for the specified delay before retrying
        time.sleep(delay / 1000)

    print(
        f"runpod-worker-comfy - Failed to connect to server at {url} after {retries} attempts."
    )
    return False


def upload_images(images):
    """
    Upload a list of base64 encoded images to the ComfyUI server using the /upload/image endpoint.

    Args:
        images (list): A list of dictionaries, each containing the 'name' of the image and the 'image' as a base64 encoded string.
        server_address (str): The address of the ComfyUI server.

    Returns:
        list: A list of responses from the server for each image upload.
    """
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []

    print(f"runpod-worker-comfy - image(s) upload")

    for image in images:
        name = image["name"]
        image_data = image["image"]
        blob = base64.b64decode(image_data)

        # Prepare the form data
        files = {
            "image": (name, BytesIO(blob), "image/png"),
            "overwrite": (None, "true"),
        }

        # POST request to upload the image
        response = requests.post(f"http://{COMFY_HOST}/upload/image", files=files)
        if response.status_code != 200:
            upload_errors.append(f"Error uploading {name}: {response.text}")
        else:
            responses.append(f"Successfully uploaded {name}")

    if upload_errors:
        print(f"runpod-worker-comfy - image(s) upload with errors")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
        }

    print(f"runpod-worker-comfy - image(s) upload complete")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
    }

def queue_workflow(workflow, client_id=None):
    """
    Queue a workflow to be processed by ComfyUI

    Args:
        workflow (dict): A dictionary containing the workflow to be processed

    Returns:
        dict: The JSON response from ComfyUI after processing the workflow
    """

    # The top level element "prompt" is required by ComfyUI
    data = {"prompt": workflow}
    if client_id is not None:
        data["client_id"] = client_id
    data = json.dumps(data).encode("utf-8")

    retries = 0
    while retries < COMFY_API_AVAILABLE_MAX_RETRIES:
        try:
            req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data)
            return json.loads(urllib.request.urlopen(req).read())
        except requests.RequestException as e:
            print(f"Error queuing workflow: {str(e)}. Retrying...")
            time.sleep(COMFY_API_AVAILABLE_INTERVAL_MS / 1000)
            retries += 1

    raise Exception("Max retries reached while queuing workflow")

def get_history(prompt_id):
    """
    Retrieve the history of a given prompt using its ID

    Args:
        prompt_id (str): The ID of the prompt whose history is to be retrieved

    Returns:
        dict: The history of the prompt, containing all the processing steps and results
    """
    with urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}") as response:
        return json.loads(response.read())

def base64_encode(file_path):
    """
    Returns base64 encoded file.

    Args:
        file_path (str): The path to the file

    Returns:
        str: The base64 encoded file
    """
    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode("utf-8")
        return f"{encoded_string}"

def base64_encode_jpeg(img_path):
    # Open the PNG image
    with Image.open(img_path) as img:
        # Convert the image to JPEG format
        with BytesIO() as output:
            img = img.convert("RGB")  # Convert to RGB mode
            img.save(output, format="JPEG", quality=85)  # Adjust quality as needed
            jpeg_data = output.getvalue()

            # Encode the JPEG image to base64
            encoded_string = base64.b64encode(jpeg_data).decode("utf-8")
            return f"{encoded_string}"

def runpod_upload_image(
    job_id,
    image_location,
    result_index=0,
    results_list=None,
    bucket_name: Optional[str] = None,
):  # pylint: disable=line-too-long # pragma: no cover
    """
    Upload a single file to bucket storage.
    """
    image_name = str(uuid.uuid4())[:8]
    boto_client, _ = rp_upload.get_boto_client()
    file_extension = os.path.splitext(image_location)[1]
    content_type = "image/" + file_extension.lstrip(".")

    with open(image_location, "rb") as input_file:
        output = input_file.read()

    if boto_client is None:
        # Save the output to a file
        print("No bucket endpoint set, saving to disk folder 'simulated_uploaded'")
        print("If this is a live endpoint, please reference the following:")
        print(
            "https://github.com/runpod/runpod-python/blob/main/docs/serverless/utils/rp_upload.md"
        )  # pylint: disable=line-too-long

        os.makedirs("simulated_uploaded", exist_ok=True)
        sim_upload_location = f"simulated_uploaded/{image_name}{file_extension}"

        with open(sim_upload_location, "wb") as file_output:
            file_output.write(output)

        if results_list is not None:
            results_list[result_index] = sim_upload_location

        return sim_upload_location

    bucket = bucket_name if bucket_name else time.strftime("%m-%y")
    boto_client.put_object(
        Bucket=f"{bucket}",
        Key=f"{job_id}/{image_name}{file_extension}",
        Body=output,
        ContentType=content_type,
    )

    presigned_url = boto_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": f"{bucket}", "Key": f"{job_id}/{image_name}{file_extension}"},
        ExpiresIn=604800,
    )

    if results_list is not None:
        results_list[result_index] = presigned_url

    return presigned_url, f"{bucket}/{job_id}/{image_name}{file_extension}"

def process_output_images(outputs, job_id, download_file_names, return_format="url"):
    """
    This function takes the "outputs" from image generation and the job ID,
    then determines the correct way to return the image, either as a direct URL
    to an AWS S3 bucket or as a base64 encoded string, depending on the
    environment configuration.

    Args:
        outputs (dict): A dictionary containing the outputs from image generation,
                        typically includes node IDs and their respective output data.
        job_id (str): The unique identifier for the job.

    Returns:
        dict: A dictionary with the status ('success' or 'error') and the message,
              which is either the URL to the image in the AWS S3 bucket or a base64
              encoded string of the image. In case of error, the message details the issue.

    The function works as follows:
    - It first determines the output path for the images from an environment variable,
      defaulting to "/comfyui/output" if not set.
    - It then iterates through the outputs to find the filenames of the generated images.
    - After confirming the existence of the image in the output folder, it checks if the
      AWS S3 bucket is configured via the BUCKET_ENDPOINT_URL environment variable.
    - If AWS S3 is configured, it uploads the image to the bucket and returns the URL.
    - If AWS S3 is not configured, it encodes the image in base64 and returns the string.
    - If the image file does not exist in the output folder, it returns an error status
      with a message indicating the missing image file.
    """

    # The path where ComfyUI stores the generated images
    COMFY_OUTPUT_PATH = os.environ.get("COMFY_OUTPUT_PATH", "/comfyui/output")
    COMFY_TEMP_PATH = os.environ.get("COMFY_TEMP_PATH", "/comfyui/temp")

    output_files = []
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for image in node_output["images"]:
                output_files.append(
                    os.path.join(image["subfolder"], image["filename"])
                )
        elif "gifs" in node_output:
            for image in node_output["gifs"]:
                output_files.append(
                    os.path.join(image["subfolder"], image["filename"])
                )

    print(f"runpod-worker-comfy - image generation is done")

    message = []
    for output_image in output_files:
        # expected image output folder
        local_image_path = f"{COMFY_OUTPUT_PATH}/{output_image}"
        local_temp_image_path = f"{COMFY_TEMP_PATH}/{output_image}"

        print(f"runpod-worker-comfy - {local_image_path}")

        # The image is in the output folder
        if os.path.exists(local_image_path):
            final_url_path = local_image_path
        elif os.path.exists(local_temp_image_path):
            final_url_path = local_temp_image_path
        else:
            print("runpod-worker-comfy - the image does not exist in the output folder")
            return {
                "status": "error",
                "message": f"the image does not exist in the specified output folder: {local_image_path}",
            }
        if return_format == "url" and os.environ.get("BUCKET_ENDPOINT_URL", False):
            # URL to image in AWS S3
            image, obj_key = runpod_upload_image(job_id, final_url_path, bucket_name="runpod-temp")
            print(
                "runpod-worker-comfy - the image was generated and uploaded to AWS S3"
            )
            message.append({"url": image, "obj_key": obj_key})
        else:
            # base64 image
            image = base64_encode(final_url_path)
            # image = base64_encode_jpeg(final_url_path)
            print(
                "runpod-worker-comfy - the image was generated and converted to base64"
            )

            message.append(image)

    download_files = []
    for file_name in download_file_names:
        local_file_path = f"{COMFY_OUTPUT_PATH}/{file_name}"
        download_files.append(base64_encode(local_file_path))

    return {
        "status": "success",
        "message": message,
        "download_files": download_files,
    }

def retry_loop(func, retrys=3, timeout=10, wait_time=2, *args, **kwargs):
    fail_messages = []
    for attempt in range(retrys):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            fail_messages.append(str(e))
            if attempt == retrys - 1:
                raise Exception(f"Failed after {retrys} retries, every retry timeout {timeout} seconds. Last error message: {str(e)}")
            time.sleep(wait_time)  # Wait before retrying

def downlaod_image_from_url(image_url, retrys=3, timeout=10):
    """
    Download an image from a given URL and return it as a PIL Image object.

    Args:
        image_url (str): The URL of the image to download.
    Returns:
        base64 encoded string of the image.
    """
    def download():
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.content

    data = retry_loop(download, retrys=retrys, timeout=timeout)  
    return base64.b64encode(data).decode('utf-8')

def download_image_from_s3(s3_obj, retrys=3, timeout=10, temp_file_dir="/s3_tmp"):
    """
    Download an image from S3 using a presigned URL and return it as a PIL Image object.

    Args:
        obj_key (str): The S3 object key of the image to download.
    Returns:
        base64 encoded string of the image.
    """
    bucket_name, object_key = s3_obj.get("bucket_name", None), s3_obj.get("object_key", None)
    if not bucket_name or not object_key:
        raise Exception("Invalid S3 object information provided. bucket_name or object_key is missing")
    os.makedirs(temp_file_dir, exist_ok=True)

    def download():
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        return response['Body'].read()
    data = retry_loop(download, retrys=retrys, timeout=timeout)    
    return base64.b64encode(data).decode('utf-8')

def download_images(images):
    new_images = []
    if images is not None:
        for image in images:
            image_type = image.get("type", "base64")
            if image_type == "url":
                image_base64 = downlaod_image_from_url(image["image"])
                new_images.append({
                    "name": image["name"],
                    "image": image_base64,
                })
            elif image_type == "base64":
                new_images.append({
                    "name": image["name"],
                    "image": image["image"],
                })
            elif image_type == "s3":
                image_base64 = download_image_from_s3(image["image"])
                new_images.append({
                    "name": image["name"],
                    "image": image_base64,
                })
            else:
                raise ValueError(f"Unsupported image type: {image_type} in image filename {image['name']}")

    return new_images

def save_base64_image_to_file(image_base64, file_path):
    """
    Save a base64 encoded image to a file.

    Args:
        image_base64 (str): The base64 encoded image string.
        file_path (str): The path where the image will be saved.
    """
    image_data = base64.b64decode(image_base64)
    with open(file_path, "wb") as file:
        file.write(image_data)

# last_data_shared = None
# lock = threading.Lock()
# def websocket_receiver_loop(ws):
#     global last_data_shared
#     while True:
#         try:
#             out = ws.recv()  # 接收一個消息
#             with lock:
#                 out_data = json.loads(out)
#                 out_type = out_data.get("type", "")
#                 if out_type in ["progress_state"]:
#                     last_data_shared = out_data  # 更新為最新（最後）的一個
#         except websocket.WebSocketTimeoutException:
#             # 超時時不break，而是繼續循環（持續接收）
#             pass
#         except websocket.WebSocketConnectionClosedException:
#             print("WebSocket connection closed in receiver thread.")
#             break
#         except Exception as e:
#             traceback.print_exc()  # 改用print_exc來打印堆棧
#             print(f"WebSocket error: {str(e)}")
#             raise e
# def get_last_websocket_data():
#     global last_data_shared
#     with lock:
#         last_data = last_data_shared  # 安全讀取
#     return last_data

def websocket_receiver(ws):
    try:
        out = ws.recv()  # 接收一個消息
        out_data = json.loads(out)
        out_type = out_data.get("type", "")
        if out_type in ["progress_state"]:
            return out_data
    except websocket.WebSocketConnectionClosedException:
        print("WebSocket connection closed in receiver thread.")
    except websocket.WebSocketTimeoutException:
        # 超時時不break，而是繼續循環（持續接收）
        pass
    except Exception as e:
        traceback.print_exc()  # 改用print_exc來打印堆棧
        print(f"WebSocket error: {str(e)}")
    return None
def websocket_connector(client_id, max_retries=1, interval_ms=100):
    ws = websocket.WebSocket()
    retries = 0
    while retries < max_retries:
        try:
            ws.connect("ws://{}/ws?clientId={}".format(COMFY_HOST, client_id))
            ws.settimeout(0.1)
            return ws
        except Exception as e:
            print(f"Error connecting to WebSocket: {str(e)}. Retrying...")
            time.sleep(interval_ms / 1000)
            retries += 1
    raise Exception("Max retries reached while connecting to WebSocket")


def queue_comfyui(images, workflow):
    # Make sure that the ComfyUI API is available
    check_server(
        f"http://{COMFY_HOST}",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    )

    # Upload images if they exist
    upload_result = upload_images(images)

    if upload_result["status"] == "error":
        yield {"error": upload_result}
        return
    
    # create websocket client id
    client_id=str(uuid.uuid4())
    # Queue the workflow
    try:
        queued_workflow = queue_workflow(workflow, client_id=client_id)
        prompt_id = queued_workflow["prompt_id"]
        print(f"runpod-worker-comfy - queued workflow with ID {prompt_id}")
    except Exception as e:
        yield {"error": f"Error queuing workflow: {str(e)}"}
        return
    
    # start websocket receiver thread
    # receiver_thread = threading.Thread(target=websocket_receiver, args=(ws,), daemon=True)
    # receiver_thread.start()

    # Poll for completion
    print(f"runpod-worker-comfy - wait until image generation is complete")
    start_time = time.time()
    ws = None
    try:
        while time.time() - start_time < COMFY_POLLING_TIMEOUT_MS / 1000:

            # 避免進度條影響結束判斷 ws 採有連上就接收 沒有就繼續執行結束判斷
            if ws is None:
                try:
                    ws = websocket_connector(client_id)
                except Exception as e:
                    ws = None
                    print(f"Error reconnecting to WebSocket: {str(e)}. Retrying...")
            if ws is not None:
                out_data = websocket_receiver(ws)
                if out_data:
                    out_type = out_data.get("type", "")
                    if out_type in ["progress_state"]: # 解析進度資料並回傳
                        total_nodes = len(workflow)
                        completed_nodes = [node_id for node_id in out_data["data"]["nodes"] if out_data["data"]["nodes"][node_id].get("state") == "finished"]
                        running_nodes = [node_id for node_id in out_data["data"]["nodes"] if out_data["data"]["nodes"][node_id].get("state") == "running"]
                        running_nodes_maxs = [out_data["data"]["nodes"][node_id].get("max", 1) for node_id in running_nodes]
                        running_nodes_values = [out_data["data"]["nodes"][node_id].get("value", 0) for node_id in running_nodes]
                        yield {"message": {
                            "raw": out_data, 
                            "progress_1_value": len(completed_nodes),
                            "progress_1_max": total_nodes, 
                            "progress_2_value": sum(running_nodes_values) if len(running_nodes) > 0 else 0,
                            "progress_2_max": sum(running_nodes_maxs) if len(running_nodes) > 0 else 1,
                        }}

            # Exit the loop if we have found the history
            history = get_history(prompt_id)
            is_finished = False
            if prompt_id in history and history[prompt_id].get("status") and history[prompt_id].get("status").get("messages"):
                for message in history[prompt_id]["status"]["messages"]:
                    if (
                        isinstance(message, list) and
                        len(message) == 2 and
                        message[0] == "execution_success" and
                        isinstance(message[1], dict) and
                        message[1].get("prompt_id") == prompt_id
                    ):
                        is_finished = True
                        break
            
            if is_finished:
                yield {"success": history[prompt_id].get("outputs")}
                break
            
            # Wait before trying again
            # time.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
        else:
            yield {"error": "Max retries reached while waiting for image generation"}
    except Exception as e:
        stack_string = traceback.format_exc()
        print(f"Error waiting for image generation: {str(e)}\n{stack_string}")
        yield {"error": f"Error waiting for image generation: {str(e)}"}

    # Close the websocket connection
    if ws is not None:
        ws.close()

def handler(job):
    """
    The main function that handles a job of generating an image.

    This function validates the input, sends a prompt to ComfyUI for processing,
    polls ComfyUI for result, and retrieves generated images.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    job_input = job["input"]
    return_format = job_input.get("return_format", "url")

    # Make sure that the input is valid
    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    # Extract validated data
    workflow = validated_data["workflow"]
    images = validated_data.get("images")
    download_file_names = validated_data.get("download_file_names") or []

    # donload images
    new_images = download_images(images)

    # Queue the workflow and poll for completion
    progress_generator = queue_comfyui(new_images, workflow)
    for progress in progress_generator:
        if "error" in progress:
            return {"error": progress["error"]}
        elif "success" in progress:
            break
        elif "message" in progress:
            # print(f"progress: {progress}")
            runpod.serverless.progress_update(job, progress)
    progress = progress["success"]

    # Get the generated image and return it as URL in an AWS bucket or as base64
    images_result = process_output_images(progress, job["id"], download_file_names, return_format=return_format)

    result = {**images_result, "refresh_worker": REFRESH_WORKER}
    # print(f"output result: {result}")
    return result


# Start the handler only if this script is run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

    # test download_image_from_s3
    # s3_obj = {"bucket_name": "gazai", "object_key": "videos/clwyxv3h30000fh6yiygggtle/00121c91-484f-4eb7-bdf3-7ed15c7185ad/output.mp4"}
    # image_base64 = download_image_from_s3(s3_obj)
    # save_base64_image_to_file(image_base64, "output.mp4")
    
    # test downlaod_image_from_url
    # image_url = "https://www.gazai.ai/images/gazai-chan-chibi-no-bg.png"
    # image_base64 = downlaod_image_from_url(image_url)
    # save_base64_image_to_file(image_base64, "test.png")
    
    # test queue_comfyui
    # COMFY_HOST = "220.135.18.159:1120"
    # workflow_path = r"./test_resources/workflows/workflow_sdxl.json"
    # with open(workflow_path, "r") as f:
    #     workflow = json.load(f)["input"]["workflow"]
    # workflow["23"]["inputs"]["noise_seed"] = random.randint(0, 100)
    # images = []
    # for progress in queue_comfyui(images, workflow):
    #     message = progress.get("message", {})
    #     # print(progress)
    #     print(message.get("progress_1_value"), message.get("progress_1_max"), 
    #           message.get("progress_2_value"), message.get("progress_2_max"))

