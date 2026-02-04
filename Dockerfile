FROM public.ecr.aws/docker/library/ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install minimal dependencies needed for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    gpg-agent \
    git \
    wget \
    curl \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-minimal-build-12-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring_1.1-1_all.deb

# Install pip for Python 3.12 and upgrade it
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.12 get-pip.py && \
    python3.12 -m pip install --upgrade pip && \
    rm get-pip.py

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12

# Set CUDA environment for building
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"

# Install Python, git and other necessary tools
RUN pip install --upgrade uv \
    && uv venv /opt/venv

# pre install comfyui
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Setup virtual environment
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Add scripts
ADD src/extra_model_paths.yaml /comfyui/extra_model_paths.yaml
ADD src/requirements.txt /requirements.txt
ADD src/create_merge_requirement.py /create_merge_requirement.py
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
ADD src/recreate_comfyui_yaml.py ./

# Optionally copy the snapshot file
ADD *snapshot*.json /

RUN chmod +x /start.sh /restore_snapshot.sh
# Start container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories
RUN mkdir -p models/checkpoints models/vae models/clip models/unet

# # Stage 3: Final image
FROM downloader AS final

# # Copy models from stage 2 to the final image
COPY --from=base /comfyui /comfyui
COPY --from=downloader /comfyui/models /comfyui/models


# Install ComfyUI and custom nodes
RUN git clone https://github.com/gazai-io/comfy_mtb.git /comfyui/custom_nodes/comfy_mtb
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /comfyui/custom_nodes/ComfyUI-VideoHelperSuite
RUN git clone https://github.com/1038lab/ComfyUI-RMBG.git /comfyui/custom_nodes/ComfyUI-RMBG
RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git /comfyui/custom_nodes/comfyui_controlnet_aux 
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui/ /comfyui/custom_nodes/was-node-suite-comfyui
RUN git clone https://github.com/Yanick112/ComfyUI-ToSVG.git /comfyui/custom_nodes/ComfyUI-ToSVG
RUN git clone https://github.com/Visionatrix/ComfyUI-Gemini.git /comfyui/custom_nodes/ComfyUI-Gemini
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes
RUN git clone https://github.com/Clybius/ComfyUI-Extra-Samplers.git /comfyui/custom_nodes/ComfyUI-Extra-Samplers
RUN git clone https://github.com/yolain/ComfyUI-Easy-Use.git /comfyui/custom_nodes/ComfyUI-Easy-Use
RUN git clone https://github.com/cubiq/ComfyUI_essentials.git /comfyui/custom_nodes/ComfyUI_essentials
RUN git clone https://github.com/mfg637/ComfyUI-ScheduledGuider-Ext.git /comfyui/custom_nodes/ComfyUI-ScheduledGuider-Ext
RUN git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git /comfyui/custom_nodes/ComfyUI-Inpaint-CropAndStitch
RUN git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git /comfyui/custom_nodes/ComfyUI_UltimateSDUpscale
RUN git clone https://github.com/BlenderNeko/ComfyUI_TiledKSampler.git /comfyui/custom_nodes/ComfyUI_TiledKSampler
RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /comfyui/custom_nodes/ComfyUI-Custom-Scripts
RUN git clone https://github.com/ramyma/A8R8_ComfyUI_nodes.git /comfyui/custom_nodes/A8R8_ComfyUI_nodes
RUN git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui.git /comfyui/custom_nodes/masquerade-nodes-comfyui
RUN git clone https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI.git /comfyui/custom_nodes/ControlNet-LLLite-ComfyUI
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /comfyui/custom_nodes/ComfyUI-Impact-Pack
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git /comfyui/custom_nodes/ComfyUI-Impact-Subpack
RUN git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git /comfyui/custom_nodes/ComfyUI-WanAnimatePreprocess
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git /comfyui/custom_nodes/ComfyUI-WanVideoWrapper
RUN git clone https://github.com/aria1th/ComfyUI-LogicUtils.git /comfyui/custom_nodes/ComfyUI-LogicUtils
RUN git clone https://github.com/1038lab/ComfyUI-QwenVL.git /comfyui/custom_nodes/ComfyUI-QwenVL

# merge all requirements.txt to merged_requirements.txt
RUN python3 /create_merge_requirement.py

# Install merged requirements.txt
RUN uv pip install -U -r "merged_requirements.txt" --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128

# Restore the snapshot to install custom nodes
# RUN /restore_snapshot.sh

# ControlNet-LLLite-ComfyUI extra install
RUN wget -O /comfyui/custom_nodes/ControlNet-LLLite-ComfyUI/models/kohya_controllllite_xl_canny_anime.safetensors https://huggingface.co/kohya-ss/controlnet-lllite/resolve/main/controllllite_v01032064e_sdxl_canny_anime.safetensors?download=true
RUN wget -O /comfyui/custom_nodes/ControlNet-LLLite-ComfyUI/models/kohya_controllllite_xl_scribble_anime.safetensors https://huggingface.co/kohya-ss/controlnet-lllite/resolve/main/controllllite_v01032064e_sdxl_fake_scribble_anime.safetensors?download=true

# ComfyUI-Impact-Pack extra install
RUN cd /comfyui/custom_nodes/ComfyUI-Impact-Pack && \
    python install.py

# ComfyUI-Impact-Subpac extra install
RUN cd /comfyui/custom_nodes/ComfyUI-Impact-Subpack && \
    python install.py

# rename comfyui_controlnet_aux/config.example.yaml to comfyui_controlnet_aux/config.yaml
# RUN mv /comfyui/custom_nodes/comfyui_controlnet_aux/config.example.yaml /comfyui/custom_nodes/comfyui_controlnet_aux/config.yaml
# yaml annotator_ckpts_path to /workspace/storage/stable_diffusion/models/controlnet
# RUN sed -i 's|annotator_ckpts_path: .*|annotator_ckpts_path: /workspace/storage/stable_diffusion/models/controlnet|' /comfyui/custom_nodes/comfyui_controlnet_aux/config.yaml

# Start container
CMD ["/start.sh"]