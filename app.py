import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import time
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import torch
from queue import Queue
from collections import Counter
import mediapipe as mp
from PIL import Image
import os
import sys

# Import from our custom modules
from facial_emotion import ResNet50, LSTMPyTorch, pth_processing, get_box, DICT_EMO
from voice_emotion import SpeechEmotionRecognizer

# Global variables for communication between threads
emotion_queue = Queue(maxsize=100)  # Store face emotions during recording
is_listening = False  # Flag to indicate when speech recognition is active
facial_emotion_counter = Counter()  # Count emotions during listening
display_emotion = None  # Current emotion to display
stop_threads = False  # Flag to stop all threads

class EmotionDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure the grid
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        
        # Create a frame for the video feed
        self.video_frame = tk.Frame(window, bg="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Create the video canvas
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for emotions display
        self.emotion_frame = tk.Frame(window, bg="#f0f0f0")
        self.emotion_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Current emotion display
        self.emotion_label = tk.Label(self.emotion_frame, text="Detected Emotions", font=("Arial", 16, "bold"))
        self.emotion_label.pack(pady=10)
        
        # Frame for displaying face and voice emotions
        self.emotions_display = tk.Frame(self.emotion_frame, bg="#f0f0f0")
        self.emotions_display.pack(fill=tk.X, padx=10, pady=5)
        
        # Face emotion display
        self.face_emotion_frame = tk.Frame(self.emotions_display, bg="#e6e6e6", bd=2, relief=tk.GROOVE)
        self.face_emotion_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(self.face_emotion_frame, text="Face Emotion:", font=("Arial", 12)).pack(anchor=tk.W, padx=5, pady=2)
        self.face_emotion_value = tk.Label(self.face_emotion_frame, text="Detecting...", font=("Arial", 12, "bold"), fg="blue")
        self.face_emotion_value.pack(anchor=tk.W, padx=5, pady=2)
        
        # Voice emotion display
        self.voice_emotion_frame = tk.Frame(self.emotions_display, bg="#e6e6e6", bd=2, relief=tk.GROOVE)
        self.voice_emotion_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(self.voice_emotion_frame, text="Voice Emotion:", font=("Arial", 12)).pack(anchor=tk.W, padx=5, pady=2)
        self.voice_emotion_value = tk.Label(self.voice_emotion_frame, text="Waiting for speech...", font=("Arial", 12, "bold"), fg="blue")
        self.voice_emotion_value.pack(anchor=tk.W, padx=5, pady=2)
        
        # Create a frame for the chat display
        self.chat_frame = tk.Frame(window)
        self.chat_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        # Chat display
        self.chat_label = tk.Label(self.chat_frame, text="Conversation", font=("Arial", 16, "bold"))
        self.chat_label.pack(pady=5)
        
        # Create scrolled text widget for chat history
        self.chat_history = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, width=70, height=10)
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.chat_history.config(state=tk.DISABLED)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("System ready. Say something or press 'q' to quit.")
        self.status_bar = tk.Label(window, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        
        # Initialize speech recognizer
        self.speech_recognizer = SpeechEmotionRecognizer()
        
        # Start the video stream in a separate thread
        self.init_video()
        
        # Start the speech emotion detection thread
        self.speech_thread = threading.Thread(target=self.speech_emotion_thread)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        # Update GUI periodically
        self.update_gui()
        
        # Start the main loop
        self.window.mainloop()
    
    def init_video(self):
        """Initialize the video capture and models for facial emotion detection"""
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Load models
        name_backbone_model = 'FER_static_ResNet50_AffectNet.pt'
        name_LSTM_model = 'Aff-Wild2'
        
        # Initialize PyTorch models
        self.pth_backbone_model = ResNet50(7, channels=3)
        self.pth_backbone_model.load_state_dict(torch.load(name_backbone_model))
        self.pth_backbone_model.eval()
        
        self.pth_LSTM_model = LSTMPyTorch()
        self.pth_LSTM_model.load_state_dict(torch.load(f'FER_dinamic_LSTM_{name_LSTM_model}.pt'))
        self.pth_LSTM_model.eval()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize storage for LSTM features
        self.lstm_features = []
        
        # Start video loop in a separate thread
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def video_loop(self):
        """Video capture and processing loop"""
        global facial_emotion_counter, is_listening, display_emotion, stop_threads
        
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            
            while not stop_threads:
                success, frame = self.cap.read()
                if not success or frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process the frame
                frame_copy = frame.copy()
                frame_copy.flags.writeable = False
                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_copy)
                frame_copy.flags.writeable = True
                
                current_emotion = None
                
                if results.multi_face_landmarks:
                    for fl in results.multi_face_landmarks:
                        startX, startY, endX, endY = get_box(fl, self.width, self.height)
                        cur_face = frame_copy[startY:endY, startX:endX]
                        
                        # Process image and extract features
                        try:
                            cur_face = pth_processing(Image.fromarray(cur_face))
                            features = torch.nn.functional.relu(self.pth_backbone_model.extract_features(cur_face)).detach().numpy()
                            
                            # Maintain feature history for LSTM
                            if len(self.lstm_features) == 0:
                                self.lstm_features = [features]*10
                            else:
                                self.lstm_features = self.lstm_features[1:] + [features]
                            
                            # Make prediction with LSTM
                            lstm_f = torch.from_numpy(np.vstack(self.lstm_features))
                            lstm_f = torch.unsqueeze(lstm_f, 0)
                            output = self.pth_LSTM_model(lstm_f).detach().numpy()
                            
                            # Get predicted emotion class
                            cl = np.argmax(output)
                            label = DICT_EMO[cl]
                            current_emotion = label
                            
                            # Update the face emotion in the UI
                            self.window.after(0, lambda e=label: self.update_face_emotion(e))
                            
                            # If speech is being recorded, add this emotion to the counter
                            if is_listening:
                                facial_emotion_counter[label] += 1
                            
                            # Draw face bounding box
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 255), 2)
                            
                            # Add text with the emotion
                            cv2.putText(frame, label, (startX, startY - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                            
                        except Exception as e:
                            print(f"Error processing face: {e}")
                
                # Display listening status
                if is_listening:
                    cv2.putText(frame, "Listening...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert frame to RGB for tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create an ImageTk object
                try:
                    img = PIL.Image.fromarray(frame_rgb)
                    imgtk = PIL.ImageTk.PhotoImage(image=img)
                    
                    # Update the canvas with the new image
                    self.window.after(0, lambda: self.update_canvas(imgtk))
                except Exception as e:
                    print(f"Error updating video: {e}")
    
    def update_canvas(self, imgtk):
        """Update the canvas with the new image"""
        self.canvas.config(width=imgtk.width(), height=imgtk.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk  # Keep a reference
    
    def update_face_emotion(self, emotion):
        """Update the face emotion label in the UI"""
        self.face_emotion_value.config(text=emotion)
    
    def update_voice_emotion(self, emotion):
        """Update the voice emotion label in the UI"""
        self.voice_emotion_value.config(text=emotion)
    
    def speech_emotion_thread(self):
        """Thread for speech emotion detection"""
        global is_listening, facial_emotion_counter, display_emotion, stop_threads
        
        # Add initial message to chat
        self.add_to_chat("System", "🎙️ Listening for speech... Say 'stop recording' to exit.")
        
        # Start infinite loop to listen, record, and analyze until "stop recording" is detected
        while not stop_threads:
            try:
                # Reset the facial emotion counter at the start of listening
                facial_emotion_counter.clear()
                
                # Set the listening flag to true to start collecting facial emotions
                is_listening = True
                self.window.after(0, lambda: self.status_var.set("Listening for speech..."))
                
                # Listen for audio input
                self.add_to_chat("System", "⏳ Listening... (Speak now)")
                audio = self.speech_recognizer.listen_for_speech()
                
                # After listening, set the flag to false
                is_listening = False
                self.window.after(0, lambda: self.status_var.set("Processing speech..."))
                
                if audio is None:
                    continue
                    
                # Get the most common facial emotion during this speech period
                if facial_emotion_counter:
                    most_common_emotion = facial_emotion_counter.most_common(1)[0][0]
                    self.add_to_chat("System", f"👁️ Most frequent facial emotion during speech: {most_common_emotion}")
                    display_emotion = most_common_emotion
                else:
                    self.add_to_chat("System", "👁️ No facial emotions detected during speech")
                    display_emotion = "No facial emotion detected"
                
                # Recognize the spoken text
                audioText = self.speech_recognizer.recognize_speech(audio)
                if not audioText:
                    continue
                
                # Add the recognized text to the chat
                self.add_to_chat("You", audioText)
                
                # Stop the loop if "stop recording" is spoken
                if "stop recording" in audioText.lower():
                    self.add_to_chat("System", "🛑 Stop command detected. Exiting...")
                    self.window.after(0, self.on_closing)
                    break
                
                # Save the captured audio to a WAV file
                if not self.speech_recognizer.save_audio(audio):
                    continue
                
                # Perform emotion classification on the recorded audio
                detected_emotion = self.speech_recognizer.classify_emotion()
                self.window.after(0, lambda e=detected_emotion: self.update_voice_emotion(e))
                
                # Get bot response
                self.window.after(0, lambda: self.status_var.set("Getting response..."))
                bot_response = self.speech_recognizer.get_bot_response(audioText, detected_emotion, display_emotion)
                
                # Add the bot response to the chat
                self.add_to_chat("Bot", bot_response)
                
                # Convert response to speech
                self.speech_recognizer.text_to_speech(bot_response)
                
                # Reset status
                self.window.after(0, lambda: self.status_var.set("Ready. Say something or press 'q' to quit."))

            except Exception as e:
                self.add_to_chat("System", f"❗ Error in speech thread: {str(e)}")
                is_listening = False
        
        self.add_to_chat("System", "✅ Process complete. Goodbye!")
    
    def add_to_chat(self, speaker, message):
        """Add a message to the chat history"""
        def _add():
            self.chat_history.config(state=tk.NORMAL)
            if speaker == "System":
                self.chat_history.insert(tk.END, f"{message}\n", "system")
            elif speaker == "You":
                self.chat_history.insert(tk.END, f"{speaker}: {message}\n", "user")
            else:  # Bot
                self.chat_history.insert(tk.END, f"{speaker}: {message}\n", "bot")
            
            # Configure tag styles
            self.chat_history.tag_config("system", foreground="gray")
            self.chat_history.tag_config("user", foreground="blue")
            self.chat_history.tag_config("bot", foreground="green")
            
            self.chat_history.config(state=tk.DISABLED)
            self.chat_history.see(tk.END)  # Scroll to bottom
        
        self.window.after(0, _add)
    
    def update_gui(self):
        """Update GUI elements periodically"""
        # Check for keyboard input to exit
        self.window.bind("<Key>", self.key_pressed)
        
        # Schedule the next update
        self.window.after(100, self.update_gui)
    
    def key_pressed(self, event):
        """Handle keyboard input"""
        if event.char == 'q':
            self.on_closing()
    
    def on_closing(self):
        """Clean up and close the application"""
        global stop_threads
        stop_threads = True
        
        # Clean up resources
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        # Wait for threads to finish
        if hasattr(self, 'video_thread') and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        if hasattr(self, 'speech_thread') and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)
        
        # Close the window
        self.window.destroy()
        sys.exit(0)

# if __name__ == "__main__":
#     # Create the main window
#     root = tk.Tk()
#     # Set window size
#     root.geometry("1200x800")
#     # Create the application
#     app = EmotionDetectionApp(root, "Multimodal Emotion Detection")