import os
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from dotenv import load_dotenv
import pygame
load_dotenv()

# Set your ElevenLabs API key
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize the ElevenLabs client
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def text_to_speech_file(text: str) -> str:
    """
    Converts text to speech and saves the output as an MP3 file.
    """
    # Convert text to speech
    response = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",  # Example voice ID
        text=text,
        model_id="eleven_turbo_v2",  # Model for low latency
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    
    # Generate a unique filename
    save_file_path = "output.mp3"

    # Save the audio stream to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"Audio file saved at {save_file_path}")
    
    pygame.mixer.init()
    pygame.mixer.music.load(save_file_path)
    pygame.mixer.music.play()

    # Wait until the audio finishes playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    

    pygame.mixer.music.stop()
    pygame.mixer.quit()
    
    os.remove(save_file_path)

# def text_to_speech_langchain(text: str) -> str:
#     speech_file = tts.run(text)
#     tts.play(speech_file)
# Example usage
# text_to_speech_file("Hello, this is a test of the ElevenLabs API.")
