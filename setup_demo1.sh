#!/bin/bash
# setup_demo1.sh - Setup script for VDAM AI Studio Demo 1
# This script configures the environment, downloads necessary models for
# OpenVoice and SadTalker, and generates base assets for the UI.
# Run this from the root `vdam` directory as: bash demo1/setup_demo1.sh

echo ">>> Setting up VDAM AI Studio Demo 1..."

# 1. Python Dependencies
echo ">>> Installing Python Packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers diffusers accelerate gradio edge-tts scipy moviepy
pip install safetensors huggingface_hub sentencepiece protobuf

# 2. Clone Repositories
echo ">>> Cloning Repositories..."
mkdir -p demo1/checkpoints
mkdir -p demo1/assets

if [ ! -d "demo1/SadTalker" ]; then
    echo "Cloning SadTalker..."
    git clone https://github.com/OpenTalker/SadTalker.git demo1/SadTalker
    pip install -r demo1/SadTalker/requirements.txt
fi

if [ ! -d "demo1/OpenVoice" ]; then
    echo "Cloning OpenVoice..."
    git clone https://github.com/myshell-ai/OpenVoice.git demo1/OpenVoice
    pip install -r demo1/OpenVoice/requirements.txt
    # MeloTTS for OpenVoice V2
    pip install git+https://github.com/myshell-ai/MeloTTS.git
    python -m unidic download
fi

# 3. Download Checkpoints
echo ">>> Downloading OpenVoice Checkpoints..."
mkdir -p demo1/checkpoints/openvoice
if [ ! -f "demo1/checkpoints/openvoice/checkpoints_v2.zip" ]; then
    wget -q https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip -O demo1/checkpoints/openvoice/checkpoints_v2.zip
    # Use standard unzip
    unzip -o demo1/checkpoints/openvoice/checkpoints_v2.zip -d demo1/checkpoints/openvoice/
fi

echo ">>> Downloading SadTalker Checkpoints..."
if [ ! -d "demo1/SadTalker/checkpoints" ] || [ -z "$(ls -A demo1/SadTalker/checkpoints)" ]; then
    pushd demo1/SadTalker
    bash scripts/download_models.sh
    popd
fi

# 4. Generate Reference Audio Assets
echo ">>> Generating Base Audio References using edge-tts..."
edge-tts --text "Hello there! This is a predefined reference voice sample for your virtual AI studio demo. You can use it as a base for voice cloning or lip syncing." --voice en-US-AriaNeural --write-media demo1/assets/siri_female.wav
edge-tts --text "Hello there! This is a predefined reference voice sample for your virtual AI studio demo. You can use it as a base for voice cloning or lip syncing." --voice en-US-GuyNeural --write-media demo1/assets/alexa_male.wav

echo ">>> Demo 1 setup complete! You can now run: python demo1/app_demo1.py"

