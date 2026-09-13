# AI Hand Gesture Recognition

This project is a simple AI-based hand gesture recognition system using a webcam.

## Features

The system recognizes the following basic hand gestures:

- Open Hand
- Fist
- Thumbs Up
- Victory / Peace

The detected gesture and confidence score are displayed on the screen in real time.

## Technologies Used

- Python
- OpenCV
- MediaPipe Gesture Recognizer
- Webcam

## How It Works

The webcam captures live video frames. OpenCV processes the video frames, while MediaPipe Gesture Recognizer detects the hand and identifies the gesture.

The recognized gesture and confidence score are then displayed on the screen in real time.

## How to Run

1. Install the required packages:

```bash
pip install -r requirements.txt
2. Make sure gesture_recognizer.task is present in the project folder.
3. Run the program:
python hand_gesture.py
4. Show your hand in front of the webcam.
5. Try the supported gestures.
6. Press Q to quit the application.
Project Files
hand_gesture.py - Main Python program
gesture_recognizer.task - MediaPipe gesture recognition model
requirements.txt - Required Python packages
README.md - Project documentation
Project Output

The application displays the detected hand gesture and its confidence score on the webcam screen.
