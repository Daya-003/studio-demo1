#!/bin/bash
echo "Setting up Demo 1 (OpenVoice V2 + Commercial Compliant SadTalker)..."

# Note: Google Colab uses Python 3.12, but models require Python 3.10.
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev unzip wget git

# Create a virtual environment with Python 3.10 if it doesn't exist
if [ ! -d "ai_env" ]; then
    python3.10 -m venv ai_env
fi

# Activate the environment
source ai_env/bin/activate

echo "Installing Core Dependencies for Python Backend..."
pip install --upgrade pip setuptools wheel
pip install gradio torch torchvision torchaudio numpy scipy opencv-python imageio pydub

echo "Setting up OpenVoice V2..."
if [ ! -d "OpenVoice" ]; then
    git clone https://github.com/myshell-ai/OpenVoice.git
    cd OpenVoice
    pip install -e .
    cd ..
fi

echo "Installing MeloTTS..."
if [ ! -d "MeloTTS" ]; then
    git clone https://github.com/myshell-ai/MeloTTS.git
    cd MeloTTS
    pip install -e .
    python -m unidic download
    cd ..
fi

echo "Downloading OpenVoice V2 Checkpoints..."
if [ ! -d "checkpoints_v2" ]; then
    wget -q https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
    unzip -q checkpoints_v2_0417.zip
fi

echo "Checking for SadTalker module..."
if [ ! -d "SadTalker" ]; then
    echo "Cloning open-source SadTalker repository for Lip Syncing..."
    git clone https://github.com/OpenTalker/SadTalker.git
    
    cd SadTalker
    echo "Downloading SadTalker Checkpoint Models..."
    bash scripts/download_models.sh
    pip install -r requirements.txt
    cd ..
else
    echo "SadTalker exists!"
fi

echo "Environment Setup Completed Successfully!"
echo "To run Demo 1, you MUST USE the virtual environment:"
echo "!source ai_env/bin/activate && python app_demo1.py"
