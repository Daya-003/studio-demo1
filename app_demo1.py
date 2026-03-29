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

sys.path.append('demo1/OpenVoice')
sys.path.append('demo1/SadTalker')

HAS_FLUX = False
HAS_SVD = False

try:
    from diffusers import FluxPipeline
    HAS_FLUX = True
except:
    pass

try:
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import export_to_video
    HAS_SVD = True
except:
    pass

class VDAMStudio:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.flux_pipe = None
        self.svd_pipe = None
        self.temp_dir = tempfile.mkdtemp()
        
        # Ensure directories exist
        os.makedirs("demo1/assets", exist_ok=True)
        os.makedirs("demo1/checkpoints/openvoice", exist_ok=True)
        
        # Base avatars and reference audio must exist or be created
        self.avatar_dict = {
            "Female Studio Avatar": "assets/female_avatar.png",
            "Male Studio Avatar": "assets/male_avatar.png",
        }
        
        self.voice_dict = {
            "Alexa (Female)": "demo1/assets/siri_female.wav",
            "Siri (Male)": "demo1/assets/alexa_male.wav",
        }
    
    def unload_models(self):
        """Free VRAM by unloading diffusers models."""
        if self.flux_pipe is not None:
            del self.flux_pipe
            self.flux_pipe = None
        if self.svd_pipe is not None:
            del self.svd_pipe
            self.svd_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def load_flux_pipeline(self):
        if self.flux_pipe is None:
            try:
                self.unload_models()
                print("Loading FLUX pipeline...")
                self.flux_pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell", # Use schnell for free commercial use & speed
                    torch_dtype=self.dtype
                )
                self.flux_pipe = self.flux_pipe.to(self.device)
                print("FLUX pipeline loaded successfully!")
            except Exception as e:
                print(f"Error loading FLUX: {e}")
                return False
        return True
    
    def load_svd_pipeline(self):
        if self.svd_pipe is None:
            try:
                self.unload_models()
                print("Loading SVD pipeline...")
                self.svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=torch.float16, variant="fp16"
                )
                if torch.cuda.is_available():
                    self.svd_pipe.enable_model_cpu_offload()
                print("SVD pipeline loaded successfully!")
            except Exception as e:
                print(f"Error loading SVD: {e}")
                return False
        return True
    
    def generate_text_to_image(self, prompt, style="photorealistic", height=576, width=1024):
        try:
            if not HAS_FLUX:
                return None, "FLUX not installed. Please run setup_demo1.sh"
            
            if not self.load_flux_pipeline():
                return None, "Failed to load FLUX pipeline"
            
            style_prompts = {
                "photorealistic": "photorealistic, highly detailed, sharp",
                "oil-painting": "oil painting, brush strokes, artistic, vibrant colors",
                "watercolor": "watercolor, soft, flowing, artistic, delicate",
                "digital-art": "digital art, concept art, illustration, detailed, vibrant",
                "3d-render": "3D render, CGI, professional, detailed, volumetric lighting"
            }
            
            enhanced_prompt = f"{prompt}, {style_prompts.get(style, '')}"
            
            with torch.no_grad():
                image = self.flux_pipe(
                    prompt=enhanced_prompt,
                    height=height,
                    width=width,
                    guidance_scale=0.0, # Schnell uses 0.0 guidance
                    num_inference_steps=4
                ).images[0]
            
            return image, f"✓ Image generated successfully! (Style: {style})"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def generate_text_to_video(self, image_input, prompt, duration=5.0, height=576, width=1024):
        try:
            if not HAS_SVD:
                return None, "SVD not installed."
            
            if image_input is None:
                return None, "Please upload an image first"
            
            if not self.load_svd_pipeline():
                return None, "Failed to load SVD pipeline"
            
            if isinstance(image_input, str) and os.path.exists(image_input):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                return None, "Invalid image input"
            
            image = image.resize((width, height))
            num_frames = max(4, int(25 * (duration / 4.0)))
            num_frames = min(25, num_frames)
            
            with torch.no_grad():
                frames = self.svd_pipe(
                    image=image,
                    height=height,
                    width=width,
                    decode_chunk_size=8,
                    generator=torch.manual_seed(42)
                ).frames[0]
            
            video_path = os.path.join(self.temp_dir, f"video_{datetime.now().timestamp()}.mp4")
            export_to_video(frames, video_path, fps=7)
            
            return video_path, f"✓ Video generated successfully!"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def generate_lip_sync(self, avatar_type, audio_input, text_input, voice_selection, emotion="neutral", intensity=50):
        try:
            self.unload_models() # Free VRAM
            
            # 1. Resolve Avatar Image
            avatar_path = self.avatar_dict.get(avatar_type, "demo1/assets/female_avatar.png")
            if not os.path.exists(avatar_path):
                return None, f"Avatar file not found: {avatar_path}. Please re-run setup_assets or setup_demo1.", None
            
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
                ref_audio = self.voice_dict.get(voice_selection, "demo1/assets/siri_female.wav")
                if not os.path.exists(ref_audio):
                    return None, f"Reference audio not found: {ref_audio}. Please re-run setup_demo1.", None
                
                cloned_audio = os.path.join(self.temp_dir, f"cloned_{datetime.now().timestamp()}.wav")
                
                # Create an isolated python script to run OpenVoice to avoid memory leaks
                ov_script = f"""import sys, torch
sys.path.append('demo1/OpenVoice')
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

source_se, _ = se_extractor.get_se('{temp_tts}', ToneColorConverter, vad=True)
target_se, _ = se_extractor.get_se('{ref_audio}', ToneColorConverter, vad=True)

converter = ToneColorConverter('demo1/checkpoints/openvoice/checkpoints_v2/converter/config.json', device='cuda' if torch.cuda.is_available() else 'cpu')
converter.load_ckpt('demo1/checkpoints/openvoice/checkpoints_v2/converter/checkpoint.pth')
converter.convert(
    audio_src_path='{temp_tts}', src_se=source_se, tgt_se=target_se,
    output_path='{cloned_audio}', message='default'
)
"""
                ov_script_path = os.path.join(self.temp_dir, "run_ov.py")
                with open(ov_script_path, "w") as f:
                    f.write(ov_script)
                
                result = subprocess.run(["python", ov_script_path], capture_output=True, text=True)
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
            
            cmd = [
                "python", "demo1/SadTalker/inference.py",
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
        Create stunning visuals with FLUX, videos with SVD, and lip-sync avatars!
        """)
        
        with gr.Tabs():
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
                        # Made outputs File/Video where appropriate
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
            
            with gr.Tab("🖼️ Text to Image"):
                gr.Markdown("### Generate Images from Text using FLUX")
                
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
                                minimum=256,
                                maximum=1024,
                                value=576,
                                step=64
                            )
                            image_width = gr.Slider(
                                label="Width",
                                minimum=256,
                                maximum=1024,
                                value=1024,
                                step=64
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
            
            with gr.Tab("🎥 Text to Video"):
                gr.Markdown("### Generate Videos from Images using SVD")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Video Settings")
                        video_image = gr.Image(
                            label="Input Image",
                            type="filepath"
                        )
                        
                        video_prompt = gr.Textbox(
                            label="Video Description (Unused in pure SVD but good for ref)",
                            placeholder="Describe the video motion and style...",
                            lines=3
                        )
                        
                        video_duration = gr.Slider(
                            label="Duration (seconds)",
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1
                        )
                        
                        with gr.Row():
                            video_height = gr.Slider(
                                label="Height",
                                minimum=256,
                                maximum=1024,
                                value=576,
                                step=64
                            )
                            video_width = gr.Slider(
                                label="Width",
                                minimum=256,
                                maximum=1024,
                                value=1024,
                                step=64
                            )
                        
                        video_btn = gr.Button("🎬 Generate Video", variant="primary")
                    
                    with gr.Column():
                        generated_video = gr.Video(label="Generated Video Output")
                        video_status = gr.Textbox(label="Status", interactive=False)
                
                def generate_video(image, prompt, duration, height, width):
                    if image is None:
                        return None, "Please upload an image first"
                    video_path, status = studio.generate_text_to_video(
                        image, prompt, duration, height, width
                    )
                    return video_path, status
                
                video_btn.click(
                    generate_video,
                    inputs=[video_image, video_prompt, video_duration, video_height, video_width],
                    outputs=[generated_video, video_status]
                )
            
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
                ## VDAM Studio Demo
                
                ### Features:
                - **Lip Sync**: Create lip-sync videos with OpenVoice cloning + SadTalker
                - **Text to Image**: Generate stunning images from text descriptions using FLUX.1-schnell
                - **Text to Video**: Create dynamic videos from images using open-source SVD
                
                ### Models Used:
                - **OpenVoice V2**: Zero-shot high fidelity voice cloning
                - **SadTalker**: Accurate facial emotion rendering
                - **FLUX.1-schnell**: Top-tier text-to-image pipeline
                - **SVD**: Stable Video Diffusion XT
                """)
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)
