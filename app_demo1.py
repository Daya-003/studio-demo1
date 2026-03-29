import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import sys
import tempfile
import asyncio
from pathlib import Path
import time
from datetime import datetime
import subprocess
import glob

# Ensure base paths are absolute and correct no matter where you run the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(BASE_DIR, 'OpenVoice'))
sys.path.append(os.path.join(BASE_DIR, 'SadTalker'))

try:
    from diffusers import PixArtAlphaPipeline
    from diffusers import CogVideoXPipeline
    from diffusers.utils import export_to_video
except Exception as e:
    print(f"Warning: diffusers missing or incomplete. Please ensure diffusers, sentencepiece, and protobuf are installed. Error: {e}")

class VDAMStudio:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.t2i_pipe = None
        self.t2v_pipe = None
        self.temp_dir = tempfile.mkdtemp()
        
        assets_dir = os.path.join(BASE_DIR, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, "checkpoints", "openvoice"), exist_ok=True)
        
        # Using BASE_DIR for the asset dicts explicitly to fix the "file not found" errors
        self.avatar_dict = {
            "Female Studio Avatar": os.path.join(assets_dir, "female_avatar.png"),
            "Male Studio Avatar": os.path.join(assets_dir, "male_avatar.png"),
        }
        
        self.voice_dict = {
            "Alexa (Female)": os.path.join(assets_dir, "siri_female.wav"),
            "Siri (Male)": os.path.join(assets_dir, "alexa_male.wav"),
        }
    
    def unload_models(self):
        """Free VRAM by unloading diffusers models."""
        if self.t2i_pipe is not None:
            del self.t2i_pipe
            self.t2i_pipe = None
        if self.t2v_pipe is not None:
            del self.t2v_pipe
            self.t2v_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def load_pixart_pipeline(self):
        if self.t2i_pipe is None:
            try:
                self.unload_models()
                print("Loading PixArt-alpha pipeline...")
                self.t2i_pipe = PixArtAlphaPipeline.from_pretrained(
                    "PixArt-alpha/PixArt-XL-2-1024-MS",
                    torch_dtype=self.dtype
                )
                if torch.cuda.is_available():
                    self.t2i_pipe.enable_model_cpu_offload()
                print("PixArt-alpha pipeline loaded successfully!")
            except Exception as e:
                print(f"Error loading PixArt: {e}")
                return False
        return True
    
    def load_cogvideo_pipeline(self):
        if self.t2v_pipe is None:
            try:
                self.unload_models()
                print("Loading CogVideoX-2b pipeline...")
                self.t2v_pipe = CogVideoXPipeline.from_pretrained(
                    "THUDM/CogVideoX-2b",
                    torch_dtype=self.dtype
                )
                if torch.cuda.is_available():
                    self.t2v_pipe.enable_model_cpu_offload()
                print("CogVideoX pipeline loaded successfully!")
            except Exception as e:
                print(f"Error loading CogVideoX: {e}")
                return False
        return True
    
    def generate_text_to_image(self, prompt, style="photorealistic", height=1024, width=1024):
        try:
            if not self.load_pixart_pipeline():
                return None, "Failed to load PixArt pipeline"
            
            style_prompts = {
                "photorealistic": "photorealistic, highly detailed, sharp, cinematic",
                "oil-painting": "oil painting, brush strokes, artistic, vibrant colors",
                "watercolor": "watercolor, soft, flowing, artistic, delicate",
                "digital-art": "digital art, concept art, illustration, detailed, vibrant",
                "3d-render": "3D render, CGI, professional, detailed, volumetric lighting"
            }
            
            enhanced_prompt = f"{prompt}, {style_prompts.get(style, '')}"
            
            with torch.no_grad():
                image = self.t2i_pipe(
                    prompt=enhanced_prompt,
                    height=int(height),
                    width=int(width),
                    guidance_scale=4.5,
                    num_inference_steps=20
                ).images[0]
            
            return image, f"✓ Image generated successfully! (Style: {style})"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def generate_text_to_video(self, prompt):
        try:
            if not prompt or prompt.strip() == "":
                return None, "Please provide a text description for the video."
            
            if not self.load_cogvideo_pipeline():
                return None, "Failed to load CogVideoX pipeline"
            
            with torch.no_grad():
                # CogVideoX-2b uses native 480x720 generation out of the box. 
                # 50 inference steps is recommended, but 25 works perfectly fine and cuts time in half! 
                frames = self.t2v_pipe(
                    prompt=prompt,
                    num_inference_steps=25,
                    guidance_scale=6.0
                ).frames[0]
            
            video_path = os.path.join(self.temp_dir, f"video_{datetime.now().timestamp()}.mp4")
            export_to_video(frames, video_path, fps=8)
            
            return video_path, f"✓ Video generated successfully!"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def generate_lip_sync(self, avatar_type, audio_input, text_input, voice_selection, emotion="neutral", intensity=50):
        try:
            self.unload_models() # Free VRAM
            
            # 1. Resolve Avatar Image
            avatar_path = self.avatar_dict.get(avatar_type, os.path.join(BASE_DIR, "assets", "female_avatar.png"))
            if not os.path.exists(avatar_path):
                return None, f"Avatar file not found: {avatar_path}. Please check your assets folder inside {BASE_DIR}.", None
            
            # 2. Resolve Audio
            audio_path = None
            if audio_input is not None:
                # Direct Upload Mode
                if isinstance(audio_input, str) and os.path.exists(audio_input):
                    audio_path = audio_input
                elif isinstance(audio_input, tuple):
                    sample_rate, audio_data = audio_input
                    audio_path = os.path.join(self.temp_dir, f"uploaded_{datetime.now().timestamp()}.wav")
                    import soundfile as sf
                    sf.write(audio_path, audio_data, sample_rate)
            else:
                # Text-To-Speech Mode + OpenVoice Cloning
                if not text_input:
                    return None, "Please provide text for speech.", None
                
                temp_tts = os.path.join(self.temp_dir, f"base_tts_{datetime.now().timestamp()}.wav")
                
                # Step A: Base TTS via edge-tts
                import edge_tts
                async def _gen():
                    # Decide base voice roughly reflecting gender
                    v_name = "en-US-AriaNeural" if "Female" in voice_selection else "en-US-GuyNeural"
                    c = edge_tts.Communicate(text_input, v_name)
                    await c.save(temp_tts)
                asyncio.run(_gen())
                
                # Step B: Tone Color Cloning via OpenVoice
                ref_audio = self.voice_dict.get(voice_selection, os.path.join(BASE_DIR, "assets", "siri_female.wav"))
                if not os.path.exists(ref_audio):
                    return None, f"Reference audio not found: {ref_audio}. Please re-run setup_demo1.", None
                
                cloned_audio = os.path.join(self.temp_dir, f"cloned_{datetime.now().timestamp()}.wav")
                ckpt_converter = os.path.join(BASE_DIR, "checkpoints", "openvoice", "checkpoints_v2", "converter")
                
                # Isolated background script for OpenVoice (no memory leak)
                ov_script = f"""import sys, torch
sys.path.append(r"{os.path.join(BASE_DIR, 'OpenVoice')}")
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

source_se, _ = se_extractor.get_se(r'{temp_tts}', ToneColorConverter, vad=True)
target_se, _ = se_extractor.get_se(r'{ref_audio}', ToneColorConverter, vad=True)

converter = ToneColorConverter(r"{ckpt_converter}/config.json", device='cuda' if torch.cuda.is_available() else 'cpu')
converter.load_ckpt(r"{ckpt_converter}/checkpoint.pth")
converter.convert(
    audio_src_path=r'{temp_tts}', src_se=source_se, tgt_se=target_se,
    output_path=r'{cloned_audio}', message='default'
)
"""
                ov_script_path = os.path.join(self.temp_dir, "run_ov.py")
                with open(ov_script_path, "w") as f:
                    f.write(ov_script)
                
                result = subprocess.run([sys.executable, ov_script_path], capture_output=True, text=True)
                if os.path.exists(cloned_audio):
                    audio_path = cloned_audio
                else:
                    return None, f"OpenVoice Error:\n{result.stderr}", None
            
            if audio_path is None:
                return None, "Failed to resolve audio input.", None

            # 3. Generate Lip Sync Video via SadTalker
            out_dir = os.path.join(self.temp_dir, "sadtalker_out")
            emotion_map = {"neutral": 0, "happy": 10, "sad": 20, "angry": 30, "excited": 40, "surprised": 25}
            pose_style = emotion_map.get(emotion.lower(), 0)
            
            sadtalker_inf = os.path.join(BASE_DIR, "SadTalker", "inference.py")
            cmd = [
                sys.executable, sadtalker_inf,
                "--driven_audio", audio_path,
                "--source_image", avatar_path,
                "--result_dir", out_dir,
                "--still",
                "--preprocess", "full",
                "--pose_style", str(pose_style),
                "--expression_scale", str(intensity / 100.0)
            ]
            
            res = subprocess.run(cmd, capture_output=True, text=True)
            
            # Find Output Video
            videos = glob.glob(f"{out_dir}/*/*.mp4")
            if not videos:
                return None, f"SadTalker Generation Failed. Log:\n{res.stderr[-500:]}", None
                
            latest_video = max(videos, key=os.path.getmtime)
            
            return latest_video, f"✓ Lip-sync video created! (Emotion: {emotion.title()}, Intensity: {intensity}%)", audio_path
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}", None

def create_demo():
    studio = VDAMStudio()
    
    with gr.Blocks(title="VDAM Studio Demo", theme=gr.themes.Soft(primary_hue="purple")) as demo:
        gr.Markdown("""
        # 🎬 VDAM Studio Demo
        ## AI-Powered Content Creation & Animation
        Create stunning visuals with PixArt-α, videos with CogVideoX-2b, and lip-sync avatars!
        """)
        
        with gr.Tabs():
            # TAB 1: LIP SYNC
            with gr.Tab("🎬 Lip Sync"):
                gr.Markdown("### Create Lip-Sync Videos with AI Avatars")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Character Selection")
                        avatar_type = gr.Dropdown(
                            label="Select Avatar",
                            choices=list(studio.avatar_dict.keys()),
                            value="Female Studio Avatar"
                        )
                        
                        gr.Markdown("#### Audio Input")
                        audio_mode = gr.Radio(
                            label="Audio Mode",
                            choices=["Text-to-Speech", "Upload Audio"],
                            value="Text-to-Speech"
                        )
                        
                        tts_text = gr.Textbox(
                            label="Text for Speech",
                            placeholder="Enter text to convert to speech...",
                            lines=3,
                            visible=True
                        )
                        
                        voice_selection = gr.Dropdown(
                            label="Voice Identity",
                            choices=list(studio.voice_dict.keys()),
                            value="Alexa (Female)",
                            visible=True
                        )
                        
                        audio_upload = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            visible=False
                        )
                        
                        def update_audio_mode(mode):
                            is_tts = mode == "Text-to-Speech"
                            return (
                                gr.Textbox(visible=is_tts),
                                gr.Dropdown(visible=is_tts),
                                gr.Audio(visible=not is_tts)
                            )
                        
                        audio_mode.change(
                            update_audio_mode,
                            inputs=audio_mode,
                            outputs=[tts_text, voice_selection, audio_upload]
                        )
                        
                        gr.Markdown("#### Animation Settings")
                        emotion = gr.Dropdown(
                            label="Emotion",
                            choices=["Neutral", "Happy", "Sad", "Angry", "Surprised", "Excited"],
                            value="Neutral"
                        )
                        
                        intensity = gr.Slider(
                            label="Intensity (%)",
                            minimum=0,
                            maximum=100,
                            value=50,
                            step=10
                        )
                        
                        lipsync_btn = gr.Button("🚀 Generate Lip-Sync Video", variant="primary")
                    
                    with gr.Column():
                        lipsync_video = gr.Video(label="Video Preview")
                        lipsync_status = gr.Textbox(label="Status", interactive=False)
                        lipsync_audio = gr.Audio(label="Audio Output", interactive=False)
                
                def generate_lipsync(avatar, mode, text, voice, audio, emotion, intensity):
                    audio_input = audio if mode == "Upload Audio" else None
                    video_path, status, audio_path = studio.generate_lip_sync(
                        avatar, audio_input, text, voice, emotion.lower(), intensity
                    )
                    return video_path, status, audio_path
                
                lipsync_btn.click(
                    generate_lipsync,
                    inputs=[avatar_type, audio_mode, tts_text, voice_selection, audio_upload, emotion, intensity],
                    outputs=[lipsync_video, lipsync_status, lipsync_audio]
                )
            
            # TAB 2: TEXT TO IMAGE
            with gr.Tab("🖼️ Text to Image"):
                gr.Markdown("### Generate Images from Text using PixArt-α (Fast)")
                
                with gr.Row():
                    with gr.Column():
                        image_prompt = gr.Textbox(
                            label="Image Prompt",
                            placeholder="Describe the image you want to create...",
                            lines=4
                        )
                        
                        image_style = gr.Dropdown(
                            label="Style",
                            choices=["photorealistic", "oil-painting", "watercolor", "digital-art", "3d-render"],
                            value="photorealistic"
                        )
                        
                        with gr.Row():
                            image_height = gr.Slider(
                                label="Height",
                                minimum=512,
                                maximum=1024,
                                value=1024,
                                step=256
                            )
                            image_width = gr.Slider(
                                label="Width",
                                minimum=512,
                                maximum=1024,
                                value=1024,
                                step=256
                            )
                        
                        image_btn = gr.Button("✨ Generate Image", variant="primary")
                    
                    with gr.Column():
                        generated_image = gr.Image(label="Generated Image")
                        image_status = gr.Textbox(label="Status", interactive=False)
                
                def generate_image(prompt, style, height, width):
                    if not prompt or prompt.strip() == "":
                        return None, "Please enter a prompt"
                    image, status = studio.generate_text_to_image(
                        prompt, style, height, width
                    )
                    return image, status
                
                image_btn.click(
                    generate_image,
                    inputs=[image_prompt, image_style, image_height, image_width],
                    outputs=[generated_image, image_status]
                )
            
            # TAB 3: TEXT TO VIDEO
            with gr.Tab("🎥 Text to Video"):
                gr.Markdown("### Generate Videos from Text using CogVideoX-2b")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Video Settings")
                        
                        video_prompt = gr.Textbox(
                            label="Video Description (Text Prompt)",
                            placeholder="Describe the video motion and style... (e.g. A cat playing with a toy in high motion)",
                            lines=3
                        )
                        
                        video_btn = gr.Button("🎬 Generate Video", variant="primary")
                    
                    with gr.Column():
                        generated_video = gr.Video(label="Generated Video Output")
                        video_status = gr.Textbox(label="Status", interactive=False)
                
                def generate_video(prompt):
                    if not prompt:
                        return None, "Please provide a valid text prompt"
                    video_path, status = studio.generate_text_to_video(prompt)
                    return video_path, status
                
                video_btn.click(
                    generate_video,
                    inputs=[video_prompt],
                    outputs=[generated_video, video_status]
                )
                
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)
