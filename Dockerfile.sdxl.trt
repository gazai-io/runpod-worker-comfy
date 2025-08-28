# Stage 1: Base image with common dependencies
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install torch, xformers which matches to the cuda version
RUN pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install comfy-cli
RUN pip install comfy-cli

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 12.4 --nvidia --version 0.3.33 --skip-manager

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install runpod
RUN pip install runpod requests

# Go back to the root
WORKDIR /

# Add scripts
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy the snapshot file
ADD *snapshot*.json /

# Restore the snapshot to install custom nodes
RUN /restore_snapshot.sh

# Start container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base as downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE
ARG HF_TOKEN
ARG GITHUB_TOKEN

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories
RUN mkdir -p models/checkpoints models/vae models/clip models/unet models/tensorrt

# Download checkpoints/vae/LoRA to include in image based on model type
RUN hf download JoeDengUserName/SDXL_TensorRT_Collection bluePencilXL_v600_B_1_C_1_H_1024_W_1024_stat_NVIDIA GeForce RTX 4090_model.engine  --local-dir models/tensorrt
RUN hf download bluepen5805/blue_pencil-XL blue_pencil-XL-v6.0.0.safetensors --local-dir models/checkpoints
RUN hf download Comfy-Org/stable-diffusion-3.5-fp8 ./text_encoders/clip_l.safetensors --local-dir models/clip
RUN hf download Comfy-Org/stable-diffusion-3.5-fp8 ./text_encoders/clip_g.safetensors --local-dir models/clip
RUN hf download stabilityai/sdxl-vae sdxl_vae.safetensors --local-dir models/vae

RUN mv models/clip/text_encoders/clip_l.safetensors models/clip/
RUN mv models/clip/text_encoders/clip_g.safetensors models/clip/
RUN rm -rf models/clip/text_encoders

# # Stage 3: Final image
FROM base as final

# # Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

Run git clone https://github.com/joedeng-ai/ComfyUI_TensorRT_FLUX.git /comfyui/custom_nodes/ComfyUI_TensorRT_FLUX && \
    pip install -r /comfyui/custom_nodes/ComfyUI_TensorRT_FLUX/requirements.txt

run ls /comfyui/models/tensorrt
# Start container
CMD ["/start.sh"]
