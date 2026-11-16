import cv2
import os

def test_detector():
    print(f"OpenCV Version: {cv2.__version__}")
    # Check OpenCV built-in Haar Cascade for frontal face
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print(f"Haar Cascade Path: {cascade_path}, Exists: {os.path.exists(cascade_path)}")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print("CascadeClassifier initialized successfully!")

    # Check MediaPipe
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        print("MediaPipe FaceDetection initialized successfully!")
    except Exception as e:
        print(f"MediaPipe error: {e}")

if __name__ == "__main__":
    test_detector()
