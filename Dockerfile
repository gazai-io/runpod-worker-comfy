# Stage 1: Base image with common dependencies
FROM nvcr.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 as base

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
    python3-pip python3-venv git wget libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && pip install --upgrade uv --break-system-packages \
    && uv venv /opt/venv

# Setup virtual environment
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Install torch with CUDA 12.9 support (PyTorch 2.5+ / 3.0+ 官方 cu129 wheel)
RUN uv pip install torch torchvision torchaudio xformers -U --index-url https://download.pytorch.org/whl/cu130

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./
RUN uv pip install -r requirements.txt

# Go back to the root
WORKDIR /

RUN uv pip install runpod requests 

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
    uv pip install -r requirements.txt && \
    python install.py

RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack /comfyui/custom_nodes/ComfyUI-Impact-Subpack && \
    cd /comfyui/custom_nodes/ComfyUI-Impact-Subpack && \
    uv pip install -r requirements.txt && \
    python install.py

# Use Gazai's MTB node since the original one takes too long to compile due to the numba issue.
RUN git clone https://github.com/gazai-io/comfy_mtb.git /comfyui/custom_nodes/comfy_mtb && \
    cd /comfyui/custom_nodes/comfy_mtb && \
    uv pip install -r requirements.txt

RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    cd /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    uv pip install -r requirements.txt

RUN git clone https://github.com/1038lab/ComfyUI-RMBG.git /comfyui/custom_nodes/ComfyUI-RMBG && \
    cd /comfyui/custom_nodes/ComfyUI-RMBG && \
    uv pip install -r requirements.txt

RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git /comfyui/custom_nodes/comfyui_controlnet_aux && \
    cd /comfyui/custom_nodes/comfyui_controlnet_aux && \
    uv pip install -r requirements.txt

RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui/ /comfyui/custom_nodes/was-node-suite-comfyui && \
    cd /comfyui/custom_nodes/was-node-suite-comfyui && \
    uv pip install -r requirements.txt

RUN git clone https://github.com/Yanick112/ComfyUI-ToSVG.git /comfyui/custom_nodes/ComfyUI-ToSVG && \
    cd /comfyui/custom_nodes/ComfyUI-ToSVG && \
    uv pip install -r requirements.txt

RUN git clone https://github.com/Visionatrix/ComfyUI-Gemini.git /comfyui/custom_nodes/ComfyUI-Gemini && \
    cd /comfyui/custom_nodes/ComfyUI-Gemini && \
    uv pip install -r requirements.txt

RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes && \
    cd /comfyui/custom_nodes/ComfyUI-KJNodes && \
    uv pip install -r requirements.txt

RUN git clone https://github.com/Clybius/ComfyUI-Extra-Samplers.git /comfyui/custom_nodes/ComfyUI-Extra-Samplers && \
    cd /comfyui/custom_nodes/ComfyUI-Extra-Samplers && \
    uv pip install -r requirements.txt

RUN git clone https://github.com/yolain/ComfyUI-Easy-Use.git /comfyui/custom_nodes/ComfyUI-Easy-Use && \
    cd /comfyui/custom_nodes/ComfyUI-Easy-Use && \
    uv pip install -r requirements.txt
RUN git clone https://github.com/cubiq/ComfyUI_essentials.git /comfyui/custom_nodes/ComfyUI_essentials && \
    cd /comfyui/custom_nodes/ComfyUI_essentials && \
    uv pip install -r requirements.txt

RUN git clone https://github.com/mfg637/ComfyUI-ScheduledGuider-Ext.git /comfyui/custom_nodes/ComfyUI-ScheduledGuider-Ext
RUN git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git /comfyui/custom_nodes/ComfyUI-Inpaint-CropAndStitch
RUN git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git /comfyui/custom_nodes/ComfyUI_UltimateSDUpscale
RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /comfyui/custom_nodes/ComfyUI-Custom-Scripts
RUN git clone https://github.com/ramyma/A8R8_ComfyUI_nodes.git /comfyui/custom_nodes/A8R8_ComfyUI_nodes
RUN git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui.git /comfyui/custom_nodes/masquerade-nodes-comfyui



# Start container
CMD ["/start.sh"]