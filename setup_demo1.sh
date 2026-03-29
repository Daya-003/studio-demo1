#!/bin/bash
# setup_demo1.sh - Install OpenVoice + SadTalker for VDAM Studio Demo
# MIT License Compliant - Commercial Use OK

echo "🚀 Setting up VDAM Studio Demo 1: OpenVoice + SadTalker..."

# Create virtual environment
python -m venv vdam_demo1_env
source vdam_demo1_env/bin/activate  # Linux/Mac
# On Windows: vdam_demo1_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU for demo, GPU optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
pip install gradio==4.44.0
pip install librosa==0.10.1
pip install numpy==1.26.4
pip install gradio_client==1.2.0

# Install OpenVoice (MIT License)
cd /tmp
git clone https://github.com/myshell-ai/OpenVoice
cd OpenVoice
pip install -e .
pip install nltk
python -c "import nltk; nltk.download('punkt')"

# Install SadTalker (MIT License)
cd /tmp
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker
pip install -r requirements.txt
pip install onnxruntime  # CPU inference
pip install gfpgan
pip install basicsr

# Create demo directory
mkdir -p ~/vdam_studio_demo1
cd ~/vdam_studio_demo1

# Download pretrained models (MIT compliant)
echo "📥 Downloading pretrained models..."

# OpenVoice models
mkdir -p checkpoints_v2
wget -O checkpoints_v2/vocab.txt https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints_v2/vocab.txt
wget -O checkpoints_v2/config.json https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints_v2/config.json
wget -O checkpoints_v2/openvoice_2024-05-04.pth https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints_v2/openvoice_2024-05-04.pth

# SadTalker models
mkdir -p sadtalker_checkpoints
wget -O sadtalker_checkpoints/whole_body.pth https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/whole_body.pth
wget -O sadtalker_checkpoints/wav2lip_gan.pth https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/wav2lip_gan.pth

echo "✅ Setup complete! Run: cd ~/vdam_studio_demo1 && python app_demo1.py"
echo "💡 For GPU support, reinstall PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121"
