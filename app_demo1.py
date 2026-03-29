import gradio as gr
import torch
import os
import sys
import tempfile
import asyncio
import subprocess
import edge_tts
import glob
from pathlib import Path
from PIL import Image

sys.path.append('demo1/OpenVoice')
sys.path.append('demo1/SadTalker')

# --- Global State for Diffusers ---
current_loaded_model = None
active_pipeline = None

def unload_models():
    """Unload diffusers pipelines from VRAM."""
    global current_loaded_model, active_pipeline
    if active_pipeline is not None:
        print(f"Unloading {current_loaded_model} from VRAM...")
        del active_pipeline
        active_pipeline = None
        current_loaded_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_flux():
    global current_loaded_model, active_pipeline
    if current_loaded_model == "flux":
        return active_pipeline
    unload_models()
    from diffusers import FluxPipeline
    print("Loading FLUX.1-schnell...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    current_loaded_model = "flux"
    active_pipeline = pipe
    return pipe

def load_svd():
    global current_loaded_model, active_pipeline
    if current_loaded_model == "svd":
        return active_pipeline
    unload_models()
    from diffusers import StableVideoDiffusionPipeline
    print("Loading SVD...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    current_loaded_model = "svd"
    active_pipeline = pipe
    return pipe

# --- Inference Functions ---

def infer_flux(prompt):
    pipe = load_flux()
    result = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    return result

def infer_svd(image_input):
    if image_input is None:
        return None
    pipe = load_svd()
    if isinstance(image_input, str):
        image_input = Image.open(image_input).convert("RGB")
    
    # Resize for SVD
    image_input = image_input.resize((1024, 576))
    frames = pipe(image_input, decode_chunk_size=8, generator=torch.manual_seed(42)).frames[0]
    
    out_path = os.path.join(tempfile.gettempdir(), "svd_output.mp4")
    from diffusers.utils import export_to_video
    export_to_video(frames, out_path, fps=7)
    return out_path

def infer_lipsync(
    avatar_type, avatar_upload, builtin_avatar,
    audio_type, tts_text, tts_voice, tts_clone_ref, direct_audio_upload,
    emotion, intensity
):
    """Handles OpenVoice TTS generation -> SadTalker video generation via subprocesses."""
    unload_models()
    yield None, "Preparing Avatar..."

    # 1. Resolve Image
    if avatar_type == "Upload Avatar" and avatar_upload is not None:
        face_img = avatar_upload
    else:
        face_img = "demo1/assets/female_avatar.png" if builtin_avatar == "Female Studio Avatar" else "demo1/assets/male_avatar.png"
    
    if not os.path.exists(face_img):
        yield None, f"Avatar file not found: {face_img}"
        return

    # 2. Resolve Audio
    final_audio = None
    if audio_type == "Direct Audio Mode" and direct_audio_upload is not None:
        final_audio = direct_audio_upload
        yield None, "Using Direct Audio..."
    else:
        yield None, "Generating Text-to-Speech..."
        temp_tts = os.path.join(tempfile.gettempdir(), "base_tts.wav")
        v_name = "en-US-AriaNeural" if tts_voice == "Alexa (Female)" else "en-US-GuyNeural"
        
        async def _gen():
            c = edge_tts.Communicate(tts_text, v_name)
            await c.save(temp_tts)
        asyncio.run(_gen())
        final_audio = temp_tts
        
        # Zero-shot voice cloning
        if audio_type == "Reference Audio for Cloning" and tts_clone_ref is not None:
            yield None, "Cloning Voice via OpenVoice..."
            cloned_audio = os.path.join(tempfile.gettempdir(), "cloned_tts.wav")
            ov_script = f"""import sys, torch
sys.path.append('demo1/OpenVoice')
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
try:
    source_se, _ = se_extractor.get_se('{temp_tts}', ToneColorConverter, vad=True)
    target_se, _ = se_extractor.get_se('{tts_clone_ref}', ToneColorConverter, vad=True)
    ckpt = 'demo1/checkpoints/openvoice/checkpoints_v2/converter'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    converter = ToneColorConverter(f'{{ckpt}}/config.json', device=device)
    converter.load_ckpt(f'{{ckpt}}/checkpoint.pth')
    converter.convert(
        audio_src_path='{temp_tts}', src_se=source_se, tgt_se=target_se,
        output_path='{cloned_audio}', message='default'
    )
except Exception as e:
    with open('ov_err.txt', 'w') as f: f.write(str(e))
"""
            with open("demo1/temp_openvoice.py", "w") as f:
                f.write(ov_script)
            
            res = subprocess.run(["python", "demo1/temp_openvoice.py"], capture_output=True, text=True)
            if os.path.exists(cloned_audio):
                final_audio = cloned_audio
            else:
                yield None, f"OpenVoice Error: {res.stderr}\nPlease check if checkpoints exist."
                return

    # 3. Resolve SadTalker
    yield None, "Generating Video (SadTalker) - This may take a few minutes..."
    out_dir = os.path.join(tempfile.gettempdir(), "sadtalker_out")
    
    # Map emotion to pose_style rough approximate (SadTalker pose styles are 0-45)
    emotion_map = {"Neutral": 0, "Happy": 10, "Sad": 20, "Angry": 30, "Surprise": 40}
    pose_style = emotion_map.get(emotion, 0)
    
    cmd = [
        "python", "demo1/SadTalker/inference.py",
        "--driven_audio", final_audio,
        "--source_image", face_img,
        "--result_dir", out_dir,
        "--still",
        "--preprocess", "full",
        "--pose_style", str(pose_style),
        "--expression_scale", str(intensity / 100.0)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # 4. Find Output Video
    videos = glob.glob(f"{out_dir}/*/*.mp4")
    if not videos:
        yield None, f"SadTalker Generation Failed. Log:\n{res.stderr[-500:]}"
        return
        
    latest_video = max(videos, key=os.path.getmtime)
    yield latest_video, "Done!"

# --- Gradio UI Layout ---

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# VDAM AI Studio - Demo 1")
    gr.Markdown("Create highly realistic talking portraits and access text-to-image/video capabilities.")
    
    with gr.Tabs():
        # TAB 1: Avatars
        with gr.TabItem("Talking Avatar (Lip Sync)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Character Selection")
                    avatar_type = gr.Radio(["Built-in Avatars", "Upload Avatar"], value="Built-in Avatars", label="Avatar Mode")
                    builtin_avatar = gr.Dropdown(["Female Studio Avatar", "Male Studio Avatar"], value="Female Studio Avatar", label="Built-in Base")
                    avatar_upload = gr.Image(type="filepath", label="Custom Avatar Image", visible=False)
                    
                    gr.Markdown("### 2. Audio Input")
                    audio_type = gr.Radio(
                        ["Predefined Voice Sets", "Reference Audio for Cloning", "Direct Audio Mode"], 
                        value="Predefined Voice Sets", label="Audio Generation Mode"
                    )
                    
                    # TTS Inputs
                    tts_text = gr.Textbox(lines=3, label="Text to Speak", value="Welcome to the V-DAM AI Studio demo.")
                    tts_voice = gr.Dropdown(["Alexa (Female)", "Siri (Male)"], value="Alexa (Female)", label="Base Voice")
                    tts_clone_ref = gr.Audio(type="filepath", label="Reference Audio (Voice to Clone)", visible=False)
                    
                    # Direct Input
                    direct_audio_upload = gr.Audio(type="filepath", label="Direct Audio Upload (.wav/.mp3)", visible=False)
                    
                    gr.Markdown("### 3. Animation Settings")
                    emotion = gr.Dropdown(["Neutral", "Happy", "Sad", "Angry", "Surprise"], value="Neutral", label="Emotion Style")
                    intensity = gr.Slider(0, 100, value=100, step=1, label="Expression Intensity (%)")
                    
                    generate_btn = gr.Button("Generate Video", variant="primary")
                    
                with gr.Column(scale=1):
                    output_video = gr.Video(label="AI Avatar Video Output")
                    status_text = gr.Textbox(label="Status", interactive=False)
                    
            # UI Interactions
            def update_avatar_view(choice):
                return gr.update(visible=choice == "Upload Avatar"), gr.update(visible=choice == "Built-in Avatars")
            
            avatar_type.change(update_avatar_view, inputs=[avatar_type], outputs=[avatar_upload, builtin_avatar])
            
            def update_audio_view(choice):
                return (
                    gr.update(visible=choice in ["Predefined Voice Sets", "Reference Audio for Cloning"]), # text
                    gr.update(visible=choice in ["Predefined Voice Sets", "Reference Audio for Cloning"]), # base voice
                    gr.update(visible=choice == "Reference Audio for Cloning"), # target se
                    gr.update(visible=choice == "Direct Audio Mode") # direct audio
                )
            audio_type.change(
                update_audio_view, 
                inputs=[audio_type], 
                outputs=[tts_text, tts_voice, tts_clone_ref, direct_audio_upload]
            )
            
            generate_btn.click(
                infer_lipsync,
                inputs=[avatar_type, avatar_upload, builtin_avatar, audio_type, tts_text, tts_voice, tts_clone_ref, direct_audio_upload, emotion, intensity],
                outputs=[output_video, status_text]
            )

        # TAB 2: T2I
        with gr.TabItem("Text to Image (FLUX)"):
            with gr.Row():
                with gr.Column():
                    flux_prompt = gr.Textbox(lines=4, label="Image Prompt", placeholder="Describe the image you want to generate...")
                    flux_btn = gr.Button("Generate Image", variant="primary")
                with gr.Column():
                    flux_output = gr.Image(label="Generated Image")
            flux_btn.click(infer_flux, inputs=[flux_prompt], outputs=[flux_output])

        # TAB 3: T2V
        with gr.TabItem("Text/Image to Video (SVD)"):
            with gr.Row():
                with gr.Column():
                    svd_image = gr.Image(type="filepath", label="Source Image")
                    svd_btn = gr.Button("Generate Video", variant="primary")
                with gr.Column():
                    svd_output = gr.Video(label="Generated Video Output")
            svd_btn.click(infer_svd, inputs=[svd_image], outputs=[svd_output])

if __name__ == "__main__":
    print("Pre-flight check: Make sure you ran setup_demo1.sh to install dependencies!")
    demo.launch(share=True)

