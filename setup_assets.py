import asyncio
import edge_tts
import os
import shutil

def setup_assets():
    os.makedirs("demo1/assets", exist_ok=True)
    female_src = r"C:\Users\dayas\.gemini\antigravity\brain\d53fcdab-6b4d-4903-a5c0-79b17cc778e2\female_avatar_1774767085290.png"
    male_src = r"C:\Users\dayas\.gemini\antigravity\brain\d53fcdab-6b4d-4903-a5c0-79b17cc778e2\male_avatar_1774767109363.png"

    # Only copy if they exist
    if os.path.exists(female_src):
        shutil.copy(female_src, "demo1/assets/female_avatar.png")
    if os.path.exists(male_src):
        shutil.copy(male_src, "demo1/assets/male_avatar.png")

async def gen_tts():
    text = "Hello there. This is a predefined reference voice sample for your virtual AI studio demo. You can use it as a base for voice cloning or lip syncing."
    
    # Female voice
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save("demo1/assets/siri_female.wav")
    
    # Male voice
    communicate2 = edge_tts.Communicate(text, "en-US-GuyNeural")
    await communicate2.save("demo1/assets/alexa_male.wav")

if __name__ == "__main__":
    setup_assets()
    asyncio.run(gen_tts())
    print("Assets generated successfully!")
