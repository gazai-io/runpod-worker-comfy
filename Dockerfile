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

# Support for the network volume
ADD src/extra_model_paths.yaml ./

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

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories
RUN mkdir -p models/checkpoints models/vae models/clip models/unet

# Download checkpoints/vae/LoRA to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    elif [ "$MODEL_TYPE" = "sd3" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp16.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      echo "Error: Use Docker.flux1dev instead"; \
      exit 1; \
    elif [ "$MODEL_TYPE" = "anima-pencil" ]; then \
      wget -O models/checkpoints/animaPencilXL_v500.safetensors https://huggingface.co/bluepen5805/anima_pencil-XL/resolve/main/anima_pencil-XL-v5.0.0.safetensors; \
    elif [ "$MODEL_TYPE" = "blue-pencil-xl" ]; then \
      wget -O models/checkpoints/bluePencilXL_v700.safetensors https://huggingface.co/bluepen5805/blue_pencil-XL/resolve/main/blue_pencil-XL-v7.0.0.safetensors; \
    elif [ "$MODEL_TYPE" = "blue-pencil-xl_v2" ]; then \
      wget -O models/checkpoints/bluePencilXL_v200.safetensors https://huggingface.co/bluepen5805/blue_pencil-XL/resolve/main/blue_pencil-XL-v2.0.0.safetensors; \
    elif [ "$MODEL_TYPE" = "blue-pencil" ]; then \
      wget -O models/checkpoints/bluePencil_v10.safetensors "https://civitai.com/api/download/models/107812?type=Model&format=SafeTensor&size=pruned&fp=fp16" && \
      wget -O models/vae/ClearVAE_v23_sd15.safetensors "https://civitai.com/api/download/models/88156?type=Model&format=SafeTensor"; \
    elif [ "$MODEL_TYPE" = "animagine-xl" ]; then \
      wget -O models/checkpoints/animagineXLV31_v30.safetensors https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors; \
    elif [ "$MODEL_TYPE" = "chimera" ]; then \
      wget -O models/checkpoints/chimera_2.safetensors "https://civitai.com/api/download/models/611419?type=Model&format=SafeTensor&size=pruned&fp=fp16"; \
    elif [ "$MODEL_TYPE" = "pony" ]; then \
      wget -O models/checkpoints/pony_diffusion_v6_xl.safetensors "https://civitai.com/api/download/models/290640?type=Model&format=SafeTensor&size=pruned&fp=fp16" && \
      wget -O models/vae/Ponyxl_V6_vae.safetensors "https://civitai.com/api/download/models/290640?type=VAE&format=SafeTensor"; \
    elif [ "$MODEL_TYPE" = "sd3-5" ]; then \
      echo "Error: Use Docker.sd35 instead"; \
      exit 1; \
    fi

# # Stage 3: Final image
FROM base as final

# # Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

RUN git clone https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI.git /comfyui/custom_nodes/ControlNet-LLLite-ComfyUI && \
    wget -O /comfyui/custom_nodes/ControlNet-LLLite-ComfyUI/models/kohya_controllllite_xl_canny_anime.safetensors https://huggingface.co/kohya-ss/controlnet-lllite/resolve/main/controllllite_v01032064e_sdxl_canny_anime.safetensors?download=true && \
    wget -O /comfyui/custom_nodes/ControlNet-LLLite-ComfyUI/models/kohya_controllllite_xl_scribble_anime.safetensors https://huggingface.co/kohya-ss/controlnet-lllite/resolve/main/controllllite_v01032064e_sdxl_fake_scribble_anime.safetensors?download=true

RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /comfyui/custom_nodes/ComfyUI-Impact-Pack && \
    cd /comfyui/custom_nodes/ComfyUI-Impact-Pack && \
    apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    pip install -r requirements.txt && \
    python install.py

RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack /comfyui/custom_nodes/ComfyUI-Impact-Subpack && \
    cd /comfyui/custom_nodes/ComfyUI-Impact-Subpack && \
    pip install -r requirements.txt && \
    python install.py

RUN git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui.git /comfyui/custom_nodes/masquerade-nodes-comfyui

# Use Gazai's MTB node since the original one takes too long to compile due to the numba issue.
RUN git clone https://github.com/gazai-io/comfy_mtb.git /comfyui/custom_nodes/comfy_mtb && \
    cd /comfyui/custom_nodes/comfy_mtb && \
    pip install -r requirements.txt

RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    cd /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt

RUN git clone https://github.com/ramyma/A8R8_ComfyUI_nodes.git /comfyui/custom_nodes/A8R8_ComfyUI_nodes

RUN git clone https://github.com/1038lab/ComfyUI-RMBG.git /comfyui/custom_nodes/ComfyUI-RMBG && \
    cd /comfyui/custom_nodes/ComfyUI-RMBG && \
    pip install -r requirements.txt

RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git /comfyui/custom_nodes/comfyui_controlnet_aux && \
    cd /comfyui/custom_nodes/comfyui_controlnet_aux && \
    pip install -r requirements.txt

RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /comfyui/custom_nodes/ComfyUI-Custom-Scripts && \
    cd /comfyui/custom_nodes/ComfyUI-Custom-Scripts

RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui/ /comfyui/custom_nodes/was-node-suite-comfyui && \
    cd /comfyui/custom_nodes/was-node-suite-comfyui && \
    pip install -r requirements.txt

RUN git clone https://github.com/Yanick112/ComfyUI-ToSVG.git /comfyui/custom_nodes/ComfyUI-ToSVG && \
    cd /comfyui/custom_nodes/ComfyUI-ToSVG && \
    pip install -r requirements.txt

# Start container
CMD ["/start.sh"]
