import threading
import time
import cv2
import numpy as np
import torch
from queue import Queue
from collections import Counter
import mediapipe as mp
from PIL import Image

# Import from our custom modules
from facial_emotion import ResNet50, LSTMPyTorch, pth_processing, get_box, display_FPS, DICT_EMO
from voice_emotion import SpeechEmotionRecognizer

# Global variables for communication between threads
emotion_queue = Queue(maxsize=100)  # Store face emotions during recording
is_listening = False  # Flag to indicate when speech recognition is active
facial_emotion_counter = Counter()  # Count emotions during listening
display_emotion = None  # Current emotion to display

# Thread for facial emotion detection
def facial_emotion_thread():
    global is_listening, facial_emotion_counter, display_emotion
    
    # Initialize MediaPipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    # Load models
    name_backbone_model = 'FER_static_ResNet50_AffectNet.pt'
    name_LSTM_model = 'Aff-Wild2'
    
    # Initialize PyTorch models
    pth_backbone_model = ResNet50(7, channels=3)
    pth_backbone_model.load_state_dict(torch.load(name_backbone_model))
    pth_backbone_model.eval()
    
    pth_LSTM_model = LSTMPyTorch()
    pth_LSTM_model.load_state_dict(torch.load(f'FER_dinamic_LSTM_{name_LSTM_model}.pt'))
    pth_LSTM_model.eval()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = np.round(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup video writer
    path_save_video = 'result.mp4'
    vid_writer = cv2.VideoWriter(path_save_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    # Initialize storage for LSTM features
    lstm_features = []
       
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            t1 = time.time()
            success, frame = cap.read()
            if frame is None:
                break
                
            frame_copy = frame.copy()
            frame_copy.flags.writeable = False
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_copy)
            frame_copy.flags.writeable = True
            
            current_emotion = None
            
            if results.multi_face_landmarks:
                for fl in results.multi_face_landmarks:
                    startX, startY, endX, endY = get_box(fl, w, h)
                    cur_face = frame_copy[startY:endY, startX:endX]
                   
                    # Process image and extract features
                    try:
                        cur_face = pth_processing(Image.fromarray(cur_face))
                        features = torch.nn.functional.relu(pth_backbone_model.extract_features(cur_face)).detach().numpy()
                        
                        # Maintain feature history for LSTM
                        if len(lstm_features) == 0:
                            lstm_features = [features]*10
                        else:
                            lstm_features = lstm_features[1:] + [features]
                        
                        # Make prediction with LSTM
                        lstm_f = torch.from_numpy(np.vstack(lstm_features))
                        lstm_f = torch.unsqueeze(lstm_f, 0)
                        output = pth_LSTM_model(lstm_f).detach().numpy()
           
                        # Get predicted emotion class
                        cl = np.argmax(output)
                        label = DICT_EMO[cl]
                        current_emotion = label
                        
                        # If speech is being recorded, add this emotion to the counter
                        if is_listening:
                            facial_emotion_counter[label] += 1
                        
                        # Draw face bounding box only
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 255), 2)
                    except Exception as e:
                        print(f"Error processing face: {e}")
            
            # Display the current emotion that should be shown (from speech thread)
            if display_emotion:
                cv2.putText(frame, f"Detected emotion during speech: {display_emotion}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display if currently listening
            if is_listening:
                cv2.putText(frame, "Listening for speech...", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            t2 = time.time()
            # Display FPS on frame
            frame = display_FPS(frame, 'FPS: {0:.1f}'.format(1 / (t2 - t1)), box_scale=.5)
            
            # Write to video and display
            vid_writer.write(frame)
            cv2.imshow('Multimodal Emotion Detection', frame)
           
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        vid_writer.release()
        cap.release()
        cv2.destroyAllWindows()

# Thread for speech emotion detection
def speech_emotion_thread():
    global is_listening, facial_emotion_counter, display_emotion
    
    # Initialize speech emotion recognizer
    speech_recognizer = SpeechEmotionRecognizer()
    
    print("🎙️ Listening for speech... Say 'stop recording' to exit.")
    
    # Start infinite loop to listen, record, and analyze until "stop recording" is detected
    while True:
        try:
            # Reset the facial emotion counter at the start of listening
            facial_emotion_counter.clear()
            
            # Set the listening flag to true to start collecting facial emotions
            is_listening = True
            
            # Listen for audio input
            audio = speech_recognizer.listen_for_speech()
            
            # After listening, set the flag to false
            is_listening = False
            
            if audio is None:
                continue
                
            # Get the most common facial emotion during this speech period
            if facial_emotion_counter:
                most_common_emotion = facial_emotion_counter.most_common(1)[0][0]
                print(f"👁️ Most frequent facial emotion during speech: {most_common_emotion}")
                display_emotion = most_common_emotion
            else:
                print("👁️ No facial emotions detected during speech")
                display_emotion = "No facial emotion detected"
            
            # Recognize the spoken text
            audioText = speech_recognizer.recognize_speech(audio)
            if not audioText:
                continue
                
            # Stop the loop if "stop recording" is spoken
            if "stop recording" in audioText.lower():
                print("🛑 Stop command detected. Exiting...")
                break
            
            # Save the captured audio to a WAV file
            if not speech_recognizer.save_audio(audio):
                continue
            
            # Perform emotion classification on the recorded audio
            detected_emotion = speech_recognizer.classify_emotion()
            
            # Get bot response
            bot_response = speech_recognizer.get_bot_response(audioText, detected_emotion, most_common_emotion)
            
            # Convert response to speech
            speech_recognizer.text_to_speech(bot_response)

        except Exception as e:
            print(f"❗ Error in speech thread: {str(e)}")
            is_listening = False
    
    print("✅ Process complete. Goodbye!")