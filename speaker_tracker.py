import os
import cv2
import numpy as np
from pathlib import Path

class SpeakerTracker:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.frontal_cascade_path = str(self.models_dir / "haarcascade_frontalface_default.xml")
        self.profile_cascade_path = str(self.models_dir / "haarcascade_profileface.xml")
        
        if not os.path.exists(self.frontal_cascade_path):
            raise FileNotFoundError(f"Cascade not found at {self.frontal_cascade_path}")
            
        self.frontal_face = cv2.CascadeClassifier(self.frontal_cascade_path)
        self.profile_face = cv2.CascadeClassifier(self.profile_cascade_path) if os.path.exists(self.profile_cascade_path) else None

    def analyze_video_trajectory(self, video_path: str, target_aspect: float = 9/16, sample_fps: int = 6) -> list:
        """
        Analyzes video frames to track the active talking person(s).
        Returns a smoothed list of crop_x positions for the video duration.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
            
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # In vertical 9:16 crop from 16:9 (e.g. 1920x1080)
        crop_w = int(raw_h * target_aspect)
        max_x = max(0, raw_w - crop_w)
        center_default_x = max_x // 2
        
        frame_interval = max(1, int(orig_fps / sample_fps))
        sample_x_targets = []
        sample_frame_indices = []
        
        last_known_target = center_default_x
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Downsample for ultra-fast face detection
                scale_factor = 0.5
                small = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor)
                
                # Detect frontal faces
                faces = self.frontal_face.detectMultiScale(small, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
                detected_boxes = []
                for (x, y, w, h) in faces:
                    detected_boxes.append((int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)))
                    
                # If no frontal face, try profile faces
                if len(detected_boxes) == 0 and self.profile_face:
                    profiles = self.profile_face.detectMultiScale(small, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
                    for (x, y, w, h) in profiles:
                        detected_boxes.append((int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)))
                
                if len(detected_boxes) == 1:
                    # Single person in frame - center crop on this person
                    fx, fy, fw, fh = detected_boxes[0]
                    person_center_x = fx + (fw // 2)
                    target_x = person_center_x - (crop_w // 2)
                    last_known_target = max(0, min(max_x, target_x))
                elif len(detected_boxes) > 1:
                    # Multiple people in frame (e.g. Amanda & Zavala)
                    # Calculate bounding box of the group
                    min_x = min(b[0] for b in detected_boxes)
                    max_person_x = max(b[0] + b[2] for b in detected_boxes)
                    group_span = max_person_x - min_x
                    
                    if group_span <= crop_w:
                        # Both fit in vertical frame! Center on both
                        mid_x = (min_x + max_person_x) // 2
                        target_x = mid_x - (crop_w // 2)
                    else:
                        # Too far apart: prioritize largest / most central face
                        sorted_faces = sorted(detected_boxes, key=lambda b: b[2] * b[3], reverse=True)
                        fx, fy, fw, fh = sorted_faces[0]
                        target_x = (fx + fw // 2) - (crop_w // 2)
                        
                    last_known_target = max(0, min(max_x, target_x))
                    
                sample_x_targets.append(last_known_target)
                sample_frame_indices.append(frame_idx)
                
            frame_idx += 1
            
        cap.release()
        
        if not sample_x_targets:
            return [center_default_x] * total_frames
            
        # Interpolate and smooth trajectory across all frames
        # Use Exponential Moving Average (EMA) to ensure smooth camera dolly glide without jitter
        raw_interpolated = np.interp(range(total_frames), sample_frame_indices, sample_x_targets)
        
        smoothed_x = []
        cur_x = raw_interpolated[0]
        alpha = 0.05  # Smooth cinematic camera gliding factor
        
        for val in raw_interpolated:
            cur_x = cur_x * (1 - alpha) + val * alpha
            smoothed_x.append(int(max(0, min(max_x, round(cur_x)))))
            
        return smoothed_x, raw_w, raw_h, crop_w

    def get_ffmpeg_crop_filter(self, smoothed_x: list, orig_fps: float = 30.0, crop_w: int = 608, raw_h: int = 1080) -> str:
        """
        Creates an ultra-efficient FFmpeg crop filter expression.
        If movement is subtle, uses static optimal crop.
        If movement shifts between speakers, uses smooth pan expression.
        """
        if not smoothed_x:
            return f"crop={crop_w}:{raw_h}:(iw-{crop_w})/2:0"
            
        avg_x = int(np.mean(smoothed_x))
        std_x = np.std(smoothed_x)
        
        # If camera is mostly stationary, stick to optimal centered speaker crop
        if std_x < 25:
            return f"crop={crop_w}:{raw_h}:{avg_x}:0,scale=1080:1920"
            
        # If camera tracks between speakers, sample keypoints every 1.5 seconds
        step = int(orig_fps * 1.5)
        keyframes = []
        for i in range(0, len(smoothed_x), step):
            t_sec = round(i / orig_fps, 2)
            keyframes.append((t_sec, smoothed_x[i]))
            
        # Build piecewise linear interpolation expression for FFmpeg
        # x(t) transitions smoothly between keyframe timestamps
        expr = f"{keyframes[-1][1]}"
        for k in range(len(keyframes) - 2, -1, -1):
            t_start, x_start = keyframes[k]
            t_end, x_end = keyframes[k + 1]
            slope = (x_end - x_start) / max(0.01, (t_end - t_start))
            current_seg = f"({x_start}+{slope}*(t-{t_start}))"
            expr = f"if(lt(t,{t_end}),{current_seg},{expr})"
            
        return f"crop={crop_w}:{raw_h}:'{expr}':0,scale=1080:1920"

if __name__ == "__main__":
    tracker = SpeakerTracker("models")
    test_clip = "D:/kick-ai-clip-engine/clips/adinross_20261116_063325_raw.mp4"
    if os.path.exists(test_clip):
        smooth_x, rw, rh, cw = tracker.analyze_video_trajectory(test_clip)
        filt = tracker.get_ffmpeg_crop_filter(smooth_x, 30.0, cw, rh)
        print("Speaker tracking successful!")
        print(f"Frames: {len(smooth_x)}, Raw: {rw}x{rh}, CropW: {cw}")
        print(f"Generated FFmpeg Filter: {filt[:120]}...")
    else:
        print(f"Test clip not found at {test_clip}")
