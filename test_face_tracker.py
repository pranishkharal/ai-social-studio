import os

import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV not installed in this environment")


def test_detector():
    print(f"OpenCV Version: {cv2.__version__}")
    # Check OpenCV built-in Haar Cascade for frontal face
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        pytest.skip("No real OpenCV Haar cascades installed in this environment")
    print(f"Haar Cascade Path: {cascade_path}, Exists: {os.path.exists(cascade_path)}")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print("CascadeClassifier initialized successfully!")

    # Check MediaPipe (Tasks API — mediapipe >= 1.0 has no mp.solutions).
    try:
        import mediapipe as mp
        from speaker_tracker import ensure_blazeface_model
        model_path = ensure_blazeface_model("models")
        if model_path is None:
            print("MediaPipe: BlazeFace model unavailable (download failed)")
        else:
            base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=base, min_detection_confidence=0.5,
                running_mode=mp.tasks.vision.RunningMode.IMAGE)
            detector = mp.tasks.vision.FaceDetector.create_from_options(options)
            detector.close()
            print("MediaPipe FaceDetector initialized successfully!")
    except Exception as e:
        print(f"MediaPipe error: {e}")

if __name__ == "__main__":
    test_detector()
