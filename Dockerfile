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
    && uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python pip==25.3

# Install ComfyUI v0.46.0
RUN git clone --no-checkout https://github.com/comfyanonymous/ComfyUI.git /comfyui && \
    git -C /comfyui checkout --detach a277b5bf213e8a44fa34817b7770b54f86ef4cea

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
RUN git clone --no-checkout https://github.com/gazai-io/comfy_mtb.git /comfyui/custom_nodes/comfy_mtb && \
    git -C /comfyui/custom_nodes/comfy_mtb checkout --detach 8b84129b35bf07376e1fbc37b86e9493bfb5bb13
RUN git clone --no-checkout https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && \
    git -C /comfyui/custom_nodes/ComfyUI-VideoHelperSuite checkout --detach 775ae691ecbab6a6d09a2c186f95d130cf8a10d3
RUN git clone --no-checkout https://github.com/1038lab/ComfyUI-RMBG.git /comfyui/custom_nodes/ComfyUI-RMBG && \
    git -C /comfyui/custom_nodes/ComfyUI-RMBG checkout --detach f871167b325835b627e233d5792c93f3cfb79b18
RUN git clone --no-checkout https://github.com/Fannovel16/comfyui_controlnet_aux.git /comfyui/custom_nodes/comfyui_controlnet_aux && \
    git -C /comfyui/custom_nodes/comfyui_controlnet_aux checkout --detach c2655d2970d810d8a3d71deed204cdf416b66132
RUN git clone --no-checkout https://github.com/WASasquatch/was-node-suite-comfyui/ /comfyui/custom_nodes/was-node-suite-comfyui && \
    git -C /comfyui/custom_nodes/was-node-suite-comfyui checkout --detach 9b3d8a791059796bc8d1f21b446e37f33e1d28bf
RUN git clone --no-checkout https://github.com/Yanick112/ComfyUI-ToSVG.git /comfyui/custom_nodes/ComfyUI-ToSVG && \
    git -C /comfyui/custom_nodes/ComfyUI-ToSVG checkout --detach 1022ba05318dccb4fe586ce5bbd09f1f0b3c7909
RUN git clone --no-checkout https://github.com/Visionatrix/ComfyUI-Gemini.git /comfyui/custom_nodes/ComfyUI-Gemini && \
    git -C /comfyui/custom_nodes/ComfyUI-Gemini checkout --detach 1e39866b1b685d1ae8765d4fcb598ac91b1a0569
RUN git clone --no-checkout https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes && \
    git -C /comfyui/custom_nodes/ComfyUI-KJNodes checkout --detach 187868f53a7d9df007dbfe85891dab41453f7933
RUN git clone --no-checkout https://github.com/Clybius/ComfyUI-Extra-Samplers.git /comfyui/custom_nodes/ComfyUI-Extra-Samplers && \
    git -C /comfyui/custom_nodes/ComfyUI-Extra-Samplers checkout --detach 52eac1b7c847d2727e0ca93ca26d9ffd77029daa
RUN git clone --no-checkout https://github.com/cubiq/ComfyUI_essentials.git /comfyui/custom_nodes/ComfyUI_essentials && \
    git -C /comfyui/custom_nodes/ComfyUI_essentials checkout --detach 9d9f4bedfc9f0321c19faf71855e228c93bd0dc9
RUN git clone --no-checkout https://github.com/mfg637/ComfyUI-ScheduledGuider-Ext.git /comfyui/custom_nodes/ComfyUI-ScheduledGuider-Ext && \
    git -C /comfyui/custom_nodes/ComfyUI-ScheduledGuider-Ext checkout --detach 10492201ad8c3d2aeaa2fae74a92f2dc0df31581
RUN git clone --no-checkout https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git /comfyui/custom_nodes/ComfyUI-Inpaint-CropAndStitch && \
    git -C /comfyui/custom_nodes/ComfyUI-Inpaint-CropAndStitch checkout --detach 6005efc02f7745649118edd89cdd084c2b9bd3ae
RUN git clone --no-checkout https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git /comfyui/custom_nodes/ComfyUI_UltimateSDUpscale && \
    git -C /comfyui/custom_nodes/ComfyUI_UltimateSDUpscale checkout --detach a5547db9e1d07d3318bb21e9e9c474f4c1e9c8df && \
    git -C /comfyui/custom_nodes/ComfyUI_UltimateSDUpscale submodule update --init --checkout repositories/ultimate_sd_upscale
RUN git clone --no-checkout https://github.com/BlenderNeko/ComfyUI_TiledKSampler.git /comfyui/custom_nodes/ComfyUI_TiledKSampler && \
    git -C /comfyui/custom_nodes/ComfyUI_TiledKSampler checkout --detach 2fd9b05d97ecffc604c642ffbb40220b182966b2
RUN git clone --no-checkout https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /comfyui/custom_nodes/ComfyUI-Custom-Scripts && \
    git -C /comfyui/custom_nodes/ComfyUI-Custom-Scripts checkout --detach 609f3afaa74b2f88ef9ce8d939626065e3247469
RUN git clone --no-checkout https://github.com/ramyma/A8R8_ComfyUI_nodes.git /comfyui/custom_nodes/A8R8_ComfyUI_nodes && \
    git -C /comfyui/custom_nodes/A8R8_ComfyUI_nodes checkout --detach 8d617cea60f2cd9e55f3152be356141d19ad48a2
RUN git clone --no-checkout https://github.com/BadCafeCode/masquerade-nodes-comfyui.git /comfyui/custom_nodes/masquerade-nodes-comfyui && \
    git -C /comfyui/custom_nodes/masquerade-nodes-comfyui checkout --detach 432cb4d146a391b387a0cd25ace824328b5b61cf
RUN git clone --no-checkout https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI.git /comfyui/custom_nodes/ControlNet-LLLite-ComfyUI && \
    git -C /comfyui/custom_nodes/ControlNet-LLLite-ComfyUI checkout --detach 103a4457d44fe0d3be586c6d6b32ec5ccdbd0793
RUN git clone --no-checkout https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /comfyui/custom_nodes/ComfyUI-Impact-Pack && \
    git -C /comfyui/custom_nodes/ComfyUI-Impact-Pack checkout --detach c4539f99df15cc94d772aeb933ae599c8117e20f
RUN git clone --no-checkout https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git /comfyui/custom_nodes/ComfyUI-Impact-Subpack && \
    git -C /comfyui/custom_nodes/ComfyUI-Impact-Subpack checkout --detach b4bf952cde06df01eb3deff27d6dc6a9fc9d464a
RUN git clone --no-checkout https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git /comfyui/custom_nodes/ComfyUI-WanAnimatePreprocess && \
    git -C /comfyui/custom_nodes/ComfyUI-WanAnimatePreprocess checkout --detach 0e0b6a2a555625acf4d4aefb780e27d06937132f
RUN git clone --no-checkout https://github.com/kijai/ComfyUI-WanVideoWrapper.git /comfyui/custom_nodes/ComfyUI-WanVideoWrapper && \
    git -C /comfyui/custom_nodes/ComfyUI-WanVideoWrapper checkout --detach a7439db30bed0c5b5fa437d77ae8a7bb220dfd00
RUN git clone --no-checkout https://github.com/aria1th/ComfyUI-LogicUtils.git /comfyui/custom_nodes/ComfyUI-LogicUtils && \
    git -C /comfyui/custom_nodes/ComfyUI-LogicUtils checkout --detach f77c699543b8f9977b24581af66ad026d340b663
RUN git clone --no-checkout https://github.com/1038lab/ComfyUI-QwenVL.git /comfyui/custom_nodes/ComfyUI-QwenVL && \
    git -C /comfyui/custom_nodes/ComfyUI-QwenVL checkout --detach 1b67b443918801f571714bab636edc1845b7002a
RUN git clone --no-checkout https://github.com/adieyal/comfyui-dynamicprompts.git /comfyui/custom_nodes/comfyui-dynamicprompts && \
    git -C /comfyui/custom_nodes/comfyui-dynamicprompts checkout --detach af3309b20fe0633f104b0d0968b8396e1f7e6d2a
RUN git clone --no-checkout https://github.com/endman100/ComfyUI_logic_lite.git /comfyui/custom_nodes/ComfyUI_logic_lite && \
    git -C /comfyui/custom_nodes/ComfyUI_logic_lite checkout --detach 46b037547fb411e29e094eaf5a378e9dd40bb08f
RUN git clone --no-checkout https://github.com/endman100/ComfyUI-See-through.git /comfyui/custom_nodes/ComfyUI-See-through && \
    git -C /comfyui/custom_nodes/ComfyUI-See-through checkout --detach 2aedaa31d58782f06c831ca39a52e70f309ab3b9
RUN git clone --no-checkout https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter.git /comfyui/custom_nodes/ComfyUI-Anima-Ref-Adapter && \
    git -C /comfyui/custom_nodes/ComfyUI-Anima-Ref-Adapter checkout --detach 99c222e1c2da472f4b514583b2dbd1b5afb4bdd9

# merge all requirements.txt to merged_requirements.txt
RUN python3 /create_merge_requirement.py

# Install merged requirements.txt
RUN uv pip install -U -r "merged_requirements.txt" --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128

# Remove xformers to avoid ABI mismatch with torch (diffusers will use fallback attention)
RUN uv pip uninstall xformers 2>/dev/null || true

# Restore the snapshot to install custom nodes
# RUN /restore_snapshot.sh

# Both Impact installers look for this marker in the custom_nodes directory.
RUN touch /comfyui/custom_nodes/skip_download_model && \
    cd /comfyui/custom_nodes/ComfyUI-Impact-Pack && \
    python install.py

# ComfyUI-Impact-Subpack extra install
RUN cd /comfyui/custom_nodes/ComfyUI-Impact-Subpack && \
    python install.py

# rename comfyui_controlnet_aux/config.example.yaml to comfyui_controlnet_aux/config.yaml
# RUN mv /comfyui/custom_nodes/comfyui_controlnet_aux/config.example.yaml /comfyui/custom_nodes/comfyui_controlnet_aux/config.yaml
# yaml annotator_ckpts_path to /workspace/storage/stable_diffusion/models/controlnet
# RUN sed -i 's|annotator_ckpts_path: .*|annotator_ckpts_path: /workspace/storage/stable_diffusion/models/controlnet|' /comfyui/custom_nodes/comfyui_controlnet_aux/config.yaml

# Start container
CMD ["/start.sh"]
