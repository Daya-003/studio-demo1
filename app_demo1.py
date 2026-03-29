import gradio as gr
import os
import subprocess
import glob
import urllib.request
import shutil
import sys

# -------------------------------------------------------------
# Asset Management
# -------------------------------------------------------------
def download_default_assets():
    """Automatically fetch fallback images and voice files from Open Source repos"""
    os.makedirs("assets", exist_ok=True)
    assets = {
        "woman_avatar.png": "https://raw.githubusercontent.com/OpenTalker/SadTalker/main/examples/source_image/art_1.png",
        "man_avatar.png": "https://raw.githubusercontent.com/OpenTalker/SadTalker/main/examples/source_image/art_2.png",
        "alexa_female_voice.wav": "https://raw.githubusercontent.com/OpenTalker/SadTalker/main/examples/driven_audio/RD_Radio31_000.wav",
        "alexa_male_voice.wav": "https://raw.githubusercontent.com/OpenTalker/SadTalker/main/examples/driven_audio/macron.wav",
        "siri_female_voice.wav": "https://raw.githubusercontent.com/OpenTalker/SadTalker/main/examples/driven_audio/bus_chinese.wav",
        "siri_male_voice.wav": "https://raw.githubusercontent.com/OpenTalker/SadTalker/main/examples/driven_audio/japanese.wav"
    }
    for filename, url in assets.items():
        filepath = os.path.join("assets", filename)
        if not os.path.exists(filepath):
            print(f"Downloading default asset: {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")

download_default_assets()

DEFAULT_WOMAN_AVATAR = os.path.abspath("assets/woman_avatar.png")
DEFAULT_MAN_AVATAR = os.path.abspath("assets/man_avatar.png")
DEFAULT_ALEXA_FEMALE = os.path.abspath("assets/alexa_female_voice.wav")
DEFAULT_ALEXA_MALE = os.path.abspath("assets/alexa_male_voice.wav")
DEFAULT_SIRI_FEMALE = os.path.abspath("assets/siri_female_voice.wav")
DEFAULT_SIRI_MALE = os.path.abspath("assets/siri_male_voice.wav")

# -------------------------------------------------------------
# AI Pipeline Execution
# -------------------------------------------------------------
def generate_video(
    avatar_mode, builtin_avatar, custom_avatar,
    audio_mode, text_input, builtin_voice, custom_voice, direct_audio,
    emotion_type, emotion_intensity
):
    # 1. Image Resolution
    if avatar_mode == "Built-in Avatars":
        source_image = DEFAULT_WOMAN_AVATAR if "Woman" in builtin_avatar else DEFAULT_MAN_AVATAR
        if not os.path.exists(source_image): return None, f"Error: Default image {source_image} missing."
    else:
        if not custom_avatar: return None, "Error: Upload a Custom Avatar image first."
        source_image = os.path.abspath(custom_avatar)

    out_audio_path = os.path.abspath("temp_demo1_audio.wav")
    
    # 2. Audio Processing (OpenVoice vs Direct)
    if audio_mode == "Direct Audio Upload":
        if not direct_audio: return None, "Error: Upload a direct audio file."
        shutil.copy(direct_audio, out_audio_path)
    else:
        # TTS Mode (OpenVoice V2)
        if not text_input or not text_input.strip():
            return None, "Error: Enter Text-to-Speech prompt."
            
        if builtin_voice == "Alexa (Female)":
            ref_audio = DEFAULT_ALEXA_FEMALE
        elif builtin_voice == "Alexa (Male)":
            ref_audio = DEFAULT_ALEXA_MALE
        elif builtin_voice == "Siri (Female)":
            ref_audio = DEFAULT_SIRI_FEMALE
        elif builtin_voice == "Siri (Male)":
            ref_audio = DEFAULT_SIRI_MALE
        else:
            if not custom_voice: return None, "Error: Upload reference audio for cloning."
            ref_audio = os.path.abspath(custom_voice)
            
        if not os.path.exists(ref_audio): return None, f"Error: Audio file {ref_audio} missing."
            
        print(f"Generating TTS Audio with cloned voice...")
        
        # Cross-Environment Execution: Write a script to be executed by openvoice_env
        openvoice_script = f"""
import torch
import os
import sys

# Append OpenVoice and MeloTTS to path
sys.path.append(os.path.abspath('OpenVoice'))
sys.path.append(os.path.abspath('MeloTTS'))

try:
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    from melo.api import TTS
except ImportError as e:
    print(f"Critical OpenVoice Module Import Error: {{e}}")
    sys.exit(1)

try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt_converter = 'checkpoints_v2/converter'
    
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        device = "xpu"

    print("Loading Tone Color Converter...")
    tone_color_converter = ToneColorConverter(f'{{ckpt_converter}}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{{ckpt_converter}}/checkpoint.pth')

    print("Extracting Speaker Embedding from Reference Audio...", r"{ref_audio}")
    target_se, audio_name = se_extractor.get_se(r"{ref_audio}", tone_color_converter, vad=False)

    print("Synthesizing Base Audio...")
    model = TTS(language='EN_NEWEST', device=device)
    speaker_ids = model.hps.data.spk2id
    
    source_se = torch.load('checkpoints_v2/base_speakers/ses/en-newest.pth', map_location=device)
    src_path = 'temp_demo1_melo.wav'
    model.tts_to_file(r"{text_input}", speaker_ids['EN-Newest'], src_path, speed=1.0)

    print("Applying Voice Clone...")
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=r"{out_audio_path}",
        message="@MyShell"
    )
except Exception as e:
    print(f"OpenVoice Error: {{e}}")
    sys.exit(1)
"""
        with open("run_openvoice.py", "w", encoding="utf-8") as f:
            f.write(openvoice_script)
            
        openvoice_python = os.path.abspath(os.path.join("openvoice_env", "bin", "python"))
        if not os.path.exists(openvoice_python):
            # Windows fallback
            openvoice_python = os.path.abspath(os.path.join("openvoice_env", "Scripts", "python.exe"))

        try:
            print("Running isolated OpenVoice environment command...")
            subprocess.run([openvoice_python, "run_openvoice.py"], check=True)
            if not os.path.exists(out_audio_path):
                return None, "OpenVoice Output failed to generate (File not found)."
        except subprocess.CalledProcessError as e:
            return None, f"OpenVoice TTS Subprocess Error: {e}"
            
    # 3. SadTalker Lip Sync (Commercial Compliant - NO GFPGAN)
    print("Running SadTalker Pipeline (in SadTalker Environment)...")
    result_dir = os.path.abspath("./results_demo1")
    os.makedirs(result_dir, exist_ok=True)
    
    # Emotion Mapping (mimicking styles using pose interpolation IDs native to SadTalker logic)
    emotion_map = {"Happy": 10, "Sad": 20, "Angry": 30, "Surprise": 40}
    pose_style = emotion_map.get(emotion_type, 0)
    
    # Scale Intensity from 0-100% slider to SadTalker expression_scale (~1.0 is normal, 2.0 is highly exaggerated)
    exp_scale = 1.0 + (emotion_intensity / 100.0)
    
    sadtalker_path = os.path.abspath("SadTalker/inference.py")
    if not os.path.exists(sadtalker_path):
        return None, "SadTalker Inference file not found. Have you run setup_demo1.sh?"
        
    sadtalker_cmd = [
        sys.executable, sadtalker_path,
        "--driven_audio", out_audio_path,
        "--source_image", source_image,
        "--result_dir", result_dir,
        "--still",
        "--preprocess", "crop",
        # NOTE: GFPGAN REMOVED FOR COMMERCIAL COMPLIANCE
        "--pose_style", str(pose_style),
        "--expression_scale", str(exp_scale)
    ]
    
    try:
        subprocess.run(sadtalker_cmd, check=True)
        videos = glob.glob(f"{result_dir}/*/*.mp4") + glob.glob(f"{result_dir}/*.mp4")
        if not videos: return None, "SadTalker generated no video."
        latest_video = max(videos, key=os.path.getctime)
        return latest_video, "Pipeline Execution Successful!"
    except subprocess.CalledProcessError as e:
        return None, f"Video Generation Error: {e}"
    except Exception as e:
        return None, f"Runtime Error: {str(e)}"

# -------------------------------------------------------------
# Gradio UI Frontend
# -------------------------------------------------------------
with gr.Blocks(title="VDAM Demo 1: Expressive Cloner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 VDAM AI Studio - Demo 1 (Expressive Cloner)")
    gr.Markdown("**Models:** OpenVoice V2 (*Instant Voice Cloning*) + SadTalker (*Lip Sync*)")
    gr.Markdown("---")
    
    with gr.Row():
        # COLUMN 1: Avatar
        with gr.Column(scale=1):
            gr.Markdown("### 1. Character Selection")
            avatar_mode = gr.Radio(["Built-in Avatars", "Upload Custom Avatar"], label="Method", value="Built-in Avatars")
            
            # Predefined Options
            builtin_avatar = gr.Dropdown(["System Image - Woman", "System Image - Man"], label="Predefined Avatar", value="System Image - Woman")
            preview_avatar = gr.Image(value=DEFAULT_WOMAN_AVATAR, label="Preview Predefined", interactive=False, height=200)
            
            # Custom Upload
            custom_avatar = gr.Image(label="Upload Custom Image File", type="filepath", visible=False)
            
            def update_avatar_preview(selected):
                img_path = DEFAULT_WOMAN_AVATAR if "Woman" in selected else DEFAULT_MAN_AVATAR
                return gr.update(value=img_path)
            builtin_avatar.change(fn=update_avatar_preview, inputs=builtin_avatar, outputs=preview_avatar)
            
            def toggle_avatar(mode):
                is_custom = mode == "Upload Custom Avatar"
                return gr.update(visible=is_custom), gr.update(visible=not is_custom), gr.update(visible=not is_custom)
            avatar_mode.change(fn=toggle_avatar, inputs=avatar_mode, outputs=[custom_avatar, builtin_avatar, preview_avatar])
            
        # COLUMN 2: Audio
        with gr.Column(scale=1):
            gr.Markdown("### 2. Audio Input")
            audio_mode = gr.Radio(["Text-to-Speech Mode", "Direct Audio Upload"], label="Audio Mode", value="Text-to-Speech Mode")
            
            # TTS Group
            with gr.Group() as tts_group:
                text_input = gr.Textbox(label="Text Prompt", placeholder="Type what the avatar should say...", lines=2)
                gr.Markdown("Select a voice to clone (OpenVoice V2):")
                builtin_voice = gr.Radio(
                    ["Alexa (Female)", "Alexa (Male)", "Siri (Female)", "Siri (Male)", "Upload Custom reference"], 
                    label="Voice Target", 
                    value="Alexa (Female)"
                )
                
                # Hidden audio player for preset preview
                preset_voice_preview = gr.Audio(value=DEFAULT_ALEXA_FEMALE, interactive=False, label="Voice Preview")
                custom_voice = gr.Audio(label="Upload your voice (WAV/MP3)", type="filepath", visible=False)
                
                def update_voice_preview(choice):
                    is_custom = choice == "Upload Custom reference"
                    if choice == "Alexa (Female)": default_audio = DEFAULT_ALEXA_FEMALE
                    elif choice == "Alexa (Male)": default_audio = DEFAULT_ALEXA_MALE
                    elif choice == "Siri (Female)": default_audio = DEFAULT_SIRI_FEMALE
                    elif choice == "Siri (Male)": default_audio = DEFAULT_SIRI_MALE
                    else: default_audio = DEFAULT_ALEXA_FEMALE
                    return gr.update(visible=is_custom), gr.update(visible=not is_custom, value=default_audio)
                
                builtin_voice.change(fn=update_voice_preview, inputs=builtin_voice, outputs=[custom_voice, preset_voice_preview])
                
            # Direct Audio Group
            with gr.Group(visible=False) as direct_audio_group:
                direct_audio = gr.Audio(label="Upload spoken audio file", type="filepath")
                
            def toggle_audio_mode(mode):
                is_tts = (mode == "Text-to-Speech Mode")
                return gr.update(visible=is_tts), gr.update(visible=not is_tts)
            audio_mode.change(fn=toggle_audio_mode, inputs=audio_mode, outputs=[tts_group, direct_audio_group])
            
        # COLUMN 3: Emotions
        with gr.Column(scale=1):
            gr.Markdown("### 3. Animation Settings")
            emotion_type = gr.Dropdown(
                ["Happy", "Sad", "Angry", "Surprise"], 
                label="Emotion Style", 
                value="Happy",
                info="Mapping to pose styles (e.g. Happy, Sad, Angry, Surprise)"
            )
            emotion_intensity = gr.Slider(
                minimum=0, maximum=100, value=0, step=1, 
                label="Emotion Intensity (%)", 
                info="0% is normal scaling. Up to 100% exaggeration."
            )
            
            generate_btn = gr.Button("🚀 Generate Demo 1 Video", variant="primary", size="lg")
            
    with gr.Row():
        output_video = gr.Video(label="Final Generated Output")
        status_text = gr.Textbox(label="Status Logs", interactive=False)
        
    generate_btn.click(
        fn=generate_video,
        inputs=[
            avatar_mode, builtin_avatar, custom_avatar,
            audio_mode, text_input, builtin_voice, custom_voice, direct_audio,
            emotion_type, emotion_intensity
        ],
        outputs=[output_video, status_text]
    )

if __name__ == "__main__":
    print("Booting Demo 1 Server...")
    demo.queue().launch(share=True)
