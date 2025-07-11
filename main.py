import warnings
import tkinter as tk
from app import EmotionDetectionApp

# Suppress warnings
warnings.simplefilter("ignore", UserWarning)

def main():
    """
    Main function to start the multimodal emotion detection system with GUI.
    """
    # Create the main window
    root = tk.Tk()
    # Set window size
    root.geometry("1200x800")
    # Create the application
    app = EmotionDetectionApp(root, "Multimodal Emotion Detection")
    # Start the mainloop
    root.mainloop()

if __name__ == "__main__":
    main()