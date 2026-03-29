import gradio as gr
import os
import torch
import numpy as np
import tempfile
from pathlib import Path
import json
from typing import Optional, Tuple

# OpenVoice imports
import openvoice
from openvoice.api import ToneColorConverter, BaseSpeakerTTS
from openvoice.utils import preprocess

# SadTalker imports
try:
    from src.facerender.animate import AnimateFromAudio
    from src.utils.preprocess import CropAndExtract
    import src.options as options
    from src.render import Render
except ImportError:
    print("SadTalker modules not found. Run setup_demo1.sh first!")
    exit(1)

class VDAMStudioDemo1:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🤖 VDAM Studio Demo 1 initialized on {self.device}")
        
        # Initialize OpenVoice
        self.ckpt = 'checkpoints_v2/openvoice_2024-05-04.pth'
        self.device = self.device
        self.demo_name = 'demo1'
        
        self.base_speaker_tts = BaseSpeakerTTS(f'{self.demo_name}/checkpoints_v2/vocab.txt', 
                                             f'{self.demo_name}/checkpoints_v2/config.json', 
                                             self.device)
        self.base_speaker_tts.load_ckpt(self.ckpt)
        self.base_speaker_tts.to(self.device)
        
        self.tone_color_converter = ToneColorConverter(f'{self.demo_name}/checkpoints_v2/config.json', self.device)
        self.tone_color_converter.load_ckpt(self.ckpt)
        self.tone_color_converter.to(self.device)
        
        # Predefined reference speakers (MIT compliant demo voices)
        self.reference_speakers = {
            "reference_alex": "reference_speakers/alex.wav",
            "reference_siri_f": "reference_speakers/siri_female.wav", 
            "reference_siri_m": "reference_speakers/siri_male.wav",
            "reference_neural": "reference_speakers/neural_tts.wav"
        }
        
        # Emotion to SadTalker parameter mapping
        self.emotion_map = {
            "Happy": {"expression_scale": 1.2, "head_pose_scale": 0.8},
            "Sad": {"expression_scale": 0.8, "head_pose_scale": 0.6},
            "Angry": {"expression_scale": 1.4, "head_pose_scale": 1.0},
            "Surprise": {"expression_scale": 1.3, "head_pose_scale": 0.9},
            "Neutral": {"expression_scale": 1.0, "head_pose_scale": 0.7}
        }
    
    def text_to_speech(self, text: str, voice: str, reference_audio: Optional[str] = None) -> str:
        """OpenVoice Zero-shot TTS with voice cloning"""
        try:
            if reference_audio and os.path.exists(reference_audio):
                source_se = preprocess(reference_audio)
            else:
                # Use predefined voice
                ref_path = self.reference_speakers.get(voice, list(self.reference_speakers.values())[0])
                if not os.path.exists(ref_path):
                    ref_path = "reference_speakers/alex.wav"  # fallback
                source_se = preprocess(ref_path)
            
            target_se, audio = self.base_speaker_tts.tts(text, source_se, temperature=0.3)
            
            # Save audio
            output_path = tempfile.mktemp(suffix=".wav")
            self.base_speaker_tts.save_wav(audio, target_se, output_path)
            
            return output_path
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
    
    def create_lipsync_video(self, source_image: str, audio_path: str, 
                           emotion: str, intensity: float) -> str:
        """SadTalker lip sync with emotion control"""
        try:
            # Emotion parameters
            emotion_params = self.emotion_map.get(emotion, self.emotion_map["Neutral"])
            expression_scale = emotion_params["expression_scale"] * (intensity / 100.0)
            head_pose_scale = emotion_params["head_pose_scale"] * (intensity / 100.0)
            
            # SadTalker preprocessing
            opt = options.test_opt_parser()
            opt['expression_scale'] = expression_scale
            opt['head_pose_scale'] = head_pose_scale
            
            # Create temp directories
            temp_dir = Path(tempfile.mkdtemp())
            cropper = CropAndExtract(opt, self.device)
            
            # Process source image
            source_image_path = cropper.cropper(source_image, temp_dir)
            
            # Animate
            animator = AnimateFromAudio(opt, self.device)
            enhancer = Render(opt, self.device)
            
            result = animator.generate(source_image_path, audio_path, temp_dir, 
                                     preprocess='crop', crop_enhance=True)
            
            # Enhance result
            video_path = enhancer.generate_enhance(result, temp_dir, audio_path, 
                                                 preprocess='crop', crop_enhance=True)
            
            return str(video_path)
            
        except Exception as e:
            print(f"LipSync Error: {e}")
            return None

# Initialize demo
demo = VDAMStudioDemo1()

# Gradio Interface
def process_pipeline(character_image, mode, text_input, voice_select, 
                    reference_audio, uploaded_audio, emotion, intensity):
    """Main pipeline: TTS -> LipSync"""
    
    # Step 1: Generate Audio
    if mode == "Text-to-Speech":
        if not text_input.strip():
            return None, None, "❌ Please enter text for TTS"
        
        audio_path = demo.text_to_speech(text_input, voice_select, reference_audio)
        if not audio_path:
            return None, None, "❌ TTS generation failed"
        
        status = f"✅ Audio generated: {os.path.basename(audio_path)}"
        
    else:  # Direct Audio
        if not uploaded_audio:
            return None, None, "❌ Please upload audio file"
        audio_path = uploaded_audio
        status = f"✅ Audio loaded: {os.path.basename(audio_path)}"
    
    # Step 2: Lip Sync
    if not character_image:
        return None, audio_path, "❌ Please select/upload character image"
    
    video_path = demo.create_lipsync_video(character_image, audio_path, emotion, intensity)
    
    if video_path:
        return video_path, audio_path, f"🎉 Complete! {status}\n✅ Video generated successfully!"
    else:
        return None, audio_path, f"⚠️ Audio OK but LipSync failed\n{status}"

# Create Gradio UI
with gr.Blocks(title="VDAM Studio Demo 1", theme=gr.themes.Soft()) as interface:
    gr.Markdown("""
    # 🎬 **VDAM Studio Demo 1** - OpenVoice + SadTalker
    **Real-time AI Avatar with Voice Cloning & Emotional Lip Sync**
    *MIT License - Commercial Use OK*
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🧑‍🎤 **Character Selection**")
            char_img = gr.Image(label="Upload Avatar or Select Built-in", 
                              type="filepath", height=300)
            
            with gr.Row():
                avatar_male = gr.Image(value="avatars/male_avatar.jpg", 
                                     label="👨 Male Avatar", visible=True)
                avatar_female = gr.Image(value="avatars/female_avatar.jpg", 
                                       label="👩 Female Avatar", visible=True)
        
        with gr.Column(scale=2):
            gr.Markdown("## 🔊 **Audio Input**")
            
            mode = gr.Radio(["Text-to-Speech", "Direct Audio"], 
                          value="Text-to-Speech", label="Mode")
            
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="💭 Text to Speak", 
                                          placeholder="Enter your text here...", 
                                          lines=2, visible=True)
                    voice_select = gr.Dropdown(
                        choices=["alexa_female", "alexa_male", "siri_female", "siri_male", "neural"],
                        value="alexa_female", label="🎙️ Voice Style"
                    )
                    ref_audio = gr.Audio(label="🔗 Reference Audio (Voice Clone)", 
                                       type="filepath", visible=True)
                
                with gr.Column():
                    uploaded_audio = gr.Audio(label="🎵 Upload Audio File", 
                                            type="filepath", visible=False)
            
            gr.Markdown("## 😊 **Animation Settings**")
            emotion = gr.Dropdown(["Happy", "Sad", "Angry", "Surprise", "Neutral"], 
                                value="Neutral", label="Emotion")
            intensity = gr.Slider(0, 100, value=75, step=5, label="Intensity (%)")
    
    # Output Section
    with gr.Row():
        video_output = gr.Video(label="🎥 Final LipSync Video", height=400)
        audio_output = gr.Audio(label="🔊 Generated Audio")
    
    status_output = gr.Markdown("Ready to create your AI avatar!", interactive=False)
    
    # Event Handlers
    def toggle_mode(mode_val):
        vis_tts = mode_val == "Text-to-Speech"
        return gr.update(visible=vis_tts), gr.update(visible=not vis_tts)
    
    mode.change(toggle_mode, inputs=mode, outputs=[text_input, uploaded_audio])
    
    submit_btn = gr.Button("🚀 Generate AI Avatar Video", variant="primary", size="lg")
    submit_btn.click(
        process_pipeline,
        inputs=[char_img, mode, text_input, voice_select, ref_audio, 
                uploaded_audio, emotion, intensity],
        outputs=[video_output, audio_output, status_output]
    )
    
    gr.Markdown("""
    ## 📋 **Quick Start**
    1. **Upload/select character** (JPG/PNG)
    2. **Choose mode**: Text-to-Speech OR Direct Audio  
    3. **Configure voice/emotion**
    4. **Click Generate** 🎬
    
    **Tech Stack**: OpenVoice (Zero-shot TTS) + SadTalker (LipSync)
    **License**: MIT - ✅ Commercial Use OK
    """)

if __name__ == "__main__":
    print("🎬 Starting VDAM Studio Demo 1...")
    print("📱 Local: http://127.0.0.1:7860")
    print("🌐 Public: Run `gradio.app` for share link")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set True for public link
        show_error=True,
        debug=True
    )
