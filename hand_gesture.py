import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

# Gesture names ko simple names mein convert karna
GESTURE_NAMES = {
    "Open_Palm": "Open Hand",
    "Closed_Fist": "Fist",
    "Thumb_Up": "Thumbs Up",
    "Victory": "Victory / Peace"
}

# MediaPipe Gesture Recognizer setup
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam open
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam open nahi ho raha.")
    exit()

frame_count = 0

with GestureRecognizer.create_from_options(options) as recognizer:

    while True:
        success, frame = cap.read()

        if not success:
            print("Error: Webcam se frame nahi mil raha.")
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp
        frame_count += 1
        timestamp_ms = int(frame_count * 33.33)

        # Gesture recognition
        result = recognizer.recognize_for_video(
            mp_image,
            timestamp_ms
        )

        gesture_text = "Unknown Gesture"

        # Gesture result
        if result.gestures:
            top_gesture = result.gestures[0][0]

            raw_name = top_gesture.category_name
            score = top_gesture.score

            gesture_text = GESTURE_NAMES.get(
                raw_name,
                "Unknown Gesture"
            )

            # Gesture text
            cv2.putText(
                frame,
                f"Gesture: {gesture_text}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Confidence
            cv2.putText(
                frame,
                f"Confidence: {score:.2f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:
            cv2.putText(
                frame,
                "Show your hand",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

        # Hand landmarks draw karna
        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17)
                ]

                h, w, _ = frame.shape

                points = []

                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (255, 0, 0),
                        -1
                    )

                for start, end in connections:
                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 0, 0),
                        2
                    )

        # Exit instruction
        cv2.putText(
            frame,
            "Press Q to Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Display
        cv2.imshow(
            "AI Hand Gesture Recognition",
            frame
        )

        # Q press = exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()