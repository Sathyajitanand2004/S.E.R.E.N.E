import speech_recognition as sr
import wave
from speechbrain.inference.interfaces import foreign_class
from langchain_core.runnables import RunnableConfig
from text_to_speech import text_to_speech_file 
from MemoryAgent import graph

class SpeechEmotionRecognizer:
    def __init__(self):
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        
        # Configure recognizer parameters
        self.recognizer.pause_threshold = 2  # Wait 2 seconds after speech ends
        self.recognizer.energy_threshold = 300  # Detect low-volume speech better
        
        # Audio filename for saving recordings
        self.filename = "detected_speech.wav"
        
        # Load the emotion classification model using foreign_class
        self.classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier"
        )
        
        # Configuration for the language model
        self.config = RunnableConfig(recursion_limit=10, configurable={"user_id": "1","thread_id": "1"})
    
    def adjust_for_ambient_noise(self, duration=1):
        """Adjust the recognizer's energy threshold to account for ambient noise"""
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
    
    def listen_for_speech(self, timeout=None, phrase_time_limit=30):
        """Listen for audio input and return the audio data"""
        with self.mic as source:
            print("⏳ Listening... (Speak now)")
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                return audio
            except Exception as e:
                print(f"Error during listening: {str(e)}")
                return None
    
    def recognize_speech(self, audio):
        """Convert audio to text using Google Speech Recognition"""
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"📝 Speech detected: {text}")
            return text
        except sr.UnknownValueError:
            print("⚠️ Sorry, could not understand the audio.")
            return None
        except sr.RequestError:
            print("🚨 Error connecting to Google API.")
            return None
        except Exception as e:
            print(f"❗ Error during speech recognition: {str(e)}")
            return None
    
    def save_audio(self, audio):
        """Save the audio to a WAV file"""
        try:
            with wave.open(self.filename, "wb") as wf:
                wf.setnchannels(1)  # Mono channel
                wf.setsampwidth(2)  # 16-bit PCM format
                wf.setframerate(44100)  # Sample rate
                wf.writeframes(audio.get_wav_data())  # Save audio data
            print(f"💾 Recording saved as '{self.filename}'")
            return True
        except Exception as e:
            print(f"Error saving audio: {str(e)}")
            return False
    
    def classify_emotion(self):
        """Classify the emotion in the saved audio file"""
        try:
            out_prob, score, index, text_lab = self.classifier.classify_file(self.filename)
            detected_emotion = text_lab[0].capitalize()
            print(f"🎭 Detected Speech Emotion: {detected_emotion}")
            return detected_emotion
        except Exception as e:
            print(f"Error classifying emotion: {str(e)}")
            return "Unknown"
    
    def get_bot_response(self, text, speech_emotion, face_emotion):
        """Get a response from the AI agent based on the text and emotions"""
        try:
            response = graph.invoke(
                {
                    "messages": [("user", text)],
                    "emotion": [("user", f"Detected Speech Emotion: {speech_emotion}  Detected face Emotion: {face_emotion}")]
                }, 
                config=self.config
            )
            bot_response = response["messages"][-1].content
            print(f"++BOT++ '{bot_response}'")
            return bot_response
        except Exception as e:
            print(f"Error getting bot response: {str(e)}")
            return "I'm having trouble processing that right now."
    
    def text_to_speech(self, text):
        """Convert text to speech"""
        try:
            text_to_speech_file(text)
            return True
        except Exception as e:
            print(f"Error during text-to-speech: {str(e)}")
            return False