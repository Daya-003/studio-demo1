#!/bin/bash
echo "Setting up Demo 1 with STRICT ENVIRONMENT ISOLATION to prevent module errors..."

# 1. Base OS Packages required for Audio and TTS (MeloTTS needs mecab)
sudo apt-get update
sudo apt-get install -y software-properties-common mecab libmecab-dev mecab-ipadic-utf8 build-essential ffmpeg
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev unzip wget git

echo "============================================="
echo "Building Environment 1: OpenVoice V2"
echo "============================================="
if [ ! -d "openvoice_env" ]; then
    python3.10 -m venv openvoice_env
fi
source openvoice_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio

if [ ! -d "OpenVoice" ]; then
    git clone https://github.com/myshell-ai/OpenVoice.git
    cd OpenVoice
    pip install -e .
    cd ..
fi

if [ ! -d "MeloTTS" ]; then
    git clone https://github.com/myshell-ai/MeloTTS.git
    cd MeloTTS
    pip install -e .
    python -m unidic download
    cd ..
fi

if [ ! -d "checkpoints_v2" ]; then
    wget -q https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
    unzip -q checkpoints_v2_0417.zip
fi
deactivate
echo "OpenVoice Environment Built!"

echo "============================================="
echo "Building Environment 2: SadTalker UI"
echo "============================================="
if [ ! -d "sadtalker_env" ]; then
    python3.10 -m venv sadtalker_env
fi
source sadtalker_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install gradio pydub

if [ ! -d "SadTalker" ]; then
    git clone https://github.com/OpenTalker/SadTalker.git
    cd SadTalker
    bash scripts/download_models.sh
    pip install -r requirements.txt
    cd ..
fi
deactivate
echo "SadTalker Environment Built!"

echo "============================================="
echo "Setup Complete!"
echo "To run Demo 1, activate the SadTalker environment ONLY:"
echo "!source sadtalker_env/bin/activate && python app_demo1.py"
echo "============================================="
