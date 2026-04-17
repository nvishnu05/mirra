import cv2
import numpy as np
import time
import json
import os
import config
import mediapipe as mp
from ultralytics import YOLO
from collections import Counter
from deepface import DeepFace

class WeightedResultsBuffer:
    """Buffer to smooth out jitter using confidence-weighted voting."""
    def __init__(self, size=config.SMOOTHING_WINDOW_SIZE):
        self.size = size
        self.buffer = [] # Stores tuples (value, confidence)

    def add(self, value, confidence):
        if confidence < 0.1:
            return
        self.buffer.append((value, confidence))
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def clear(self):
        """Reset the buffer history."""
        self.buffer = []

    def get_stable_value(self, current_value):
        if not self.buffer:
            return current_value
        
        votes = {}
        for val, conf in self.buffer:
            v = votes.get(val, 0)
            votes[val] = v + (conf ** 2)
            
        if not votes: return current_value
        return max(votes, key=votes.get)

class MirraModule:
    def __init__(self):
        print("[INFO] Initializing Mirra (Spec-Fix Version)...")
        # YOLOv8 Face Detection (replacing MediaPipe FaceDetection)
        # Weights: yolov8n.pt (Downloaded to models/)
        try:
            self.face_detector = YOLO(config.YOLO_MODEL_PATH)
            print(f"[INFO] YOLOv8 initialized from {config.YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLOv8: {e}. Falling back to basic detection.")
            self.face_detector = None

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Image Enhancers
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # Buffers
        self.age_buffer = WeightedResultsBuffer()
        self.gender_buffer = WeightedResultsBuffer()
        self.emotion_buffer = WeightedResultsBuffer(size=config.EMOTION_SMOOTHING_WINDOW)

        self.last_detection_time = 0
        self.last_seen_time = time.time() # Track for buffer resetting
        self.latest_result = None # Stores last Gender, Age, Emotion


    def process_frame(self, frame, detection_results=None, mesh_results=None):
        """Unified analysis with Double-Validation (YOLO + FaceMesh)."""
        # 1. Validation check: Must have YOLO body AND FaceMesh landmarks
        if not detection_results or not mesh_results or not mesh_results.multi_face_landmarks:
            if time.time() - self.last_seen_time > 3.0:
                self.age_buffer.clear()
                self.gender_buffer.clear()
                self.emotion_buffer.clear()
            return {"timestamp": time.time(), "person_detected": False}

        self.last_seen_time = time.time()
        frame_h, frame_w = frame.shape[:2]
        
        # 2. Get high-precision crop from FaceMesh
        # Using specific landmark indices for top, bottom, left, right
        face_landmarks = mesh_results.multi_face_landmarks[0]
        pts = face_landmarks.landmark
        # Indices: 10 (top), 152 (bottom), 234 (left), 454 (right)
        x_min = min([p.x for p in pts]) * frame_w
        x_max = max([p.x for p in pts]) * frame_w
        y_min = min([p.y for p in pts]) * frame_h
        y_max = max([p.y for p in pts]) * frame_h
        
        # Add slight padding for expression context (15%)
        fw, fh = x_max - x_min, y_max - y_min
        fx1, fy1 = max(0, int(x_min - fw * 0.15)), max(0, int(y_min - fh * 0.15))
        fx2, fy2 = min(frame_w, int(x_max + fw * 0.15)), min(frame_h, int(y_max + fh * 0.15))
        face_roi = frame[fy1:fy2, fx1:fx2]

        # 3. Demographics & Emotion Processing (Optimized & Silent)
        if face_roi.size > 0:
            try:
                objs = DeepFace.analyze(
                    face_roi, 
                    actions=['age', 'gender', 'emotion'], 
                    enforce_detection=False,
                    detector_backend='skip', # We already have a verified face crop
                    align=True,
                    silent=config.SILENT_MODE
                )
                
                if objs:
                    res = objs[0]
                    # Map results
                    gender_val = config.GENDER_MAP.get(res['dominant_gender'], "unknown")
                    gender_conf = float(res['gender'][res['dominant_gender']]) / 100.0
                    age_numeric = int(res['age'])
                    age_val = self.get_age_category(age_numeric)
                    age_conf = 1.0 # DeepFace point estimate
                    emotion_val = config.EMOTION_MAP.get(res['dominant_emotion'], res['dominant_emotion'])
                    emotion_conf = float(res['emotion'][res['dominant_emotion']]) / 100.0
                    emotion_proportions = {config.EMOTION_MAP.get(k, k): round(float(v)/100.0, 4) for k, v in res['emotion'].items()}
                else:
                    return {"timestamp": time.time(), "person_detected": False}
            except Exception as e:
                print(f"[DEBUG] AI Error: {e}")
                return {"timestamp": time.time(), "person_detected": False}
            
            # Buffer updates for stability
            self.age_buffer.add(age_val, 1.0)
            self.gender_buffer.add(gender_val, gender_conf)
            self.emotion_buffer.add(emotion_val, emotion_conf)
            
            stable_age = self.age_buffer.get_stable_value(age_val)
            stable_gender = self.gender_buffer.get_stable_value(gender_val)
            stable_emotion = self.emotion_buffer.get_stable_value(emotion_val)

            result = {
                "timestamp": time.time(),
                "person_detected": True,
                "gender": {"value": stable_gender, "confidence": round(gender_conf, 2)},
                "age_group": {"value": stable_age, "confidence": round(age_conf, 2)},
                "emotion": {
                    "value": stable_emotion, 
                    "confidence": round(emotion_conf, 2),
                    "proportions": emotion_proportions
                },
                "people_in_frame": len(detection_results)
            }
            
            self.latest_result = result
            return result
        
        return {"timestamp": time.time(), "person_detected": False}

    def get_age_category(self, age):
        """Maps numeric age to PRD categories."""
        for category, (min_age, max_age) in config.AGE_GROUPS.items():
            if min_age <= age <= max_age:
                return category
        return "unknown"

    def visualize_frame(self, frame, detection_results=None):
        """Draws FaceMesh and YOLO Bounding Box on every frame."""
        h, w, _ = frame.shape
        
        # 1. YOLO Bounding Box Tracking & Attributes
        if config.VISUALIZE_BOUNDING_BOX and detection_results is not None:
            for det in detection_results:
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                
                # Determine color based on emotion (if available)
                box_color = (0, 255, 0) 
                attr_text = []
                
                if self.latest_result and self.latest_result["person_detected"]:
                    res = self.latest_result
                    gender = res["gender"]["value"]
                    age = res["age_group"]["value"]
                    emotion = res["emotion"]["value"]
                    
                    attr_text = [f"{gender.upper()}", f"AGE: {age.upper()}", f"EMO: {emotion.upper()}"]
                    
                    emo_colors = {
                        "happy": (0, 255, 255), "sad": (255, 0, 0), "angry": (0, 0, 255),
                        "fear": (255, 0, 255), "surprised": (0, 165, 255), "neutral": (0, 255, 0)
                    }
                    box_color = emo_colors.get(emotion, (0, 255, 0))

                # Draw YOLO Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                for i, text in enumerate(attr_text):
                    cv2.putText(frame, text, (x1, y1 - 10 - (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                
                if not attr_text:
                    cv2.putText(frame, "TRACKING (YOLO)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 2. Face Mesh Visualization
        if config.VISUALIZE_FACE_MESH:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_results = self.face_mesh.process(frame_rgb)
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    # A. Draw General Tesselation
                    self.mp_drawing.draw_landmarks(
                        image=frame, landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1)
                    )
                    # B. Highlight Expression Contours
                    self.mp_drawing.draw_landmarks(
                        image=frame, landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
                    )
                    # C. Lips & Eyes
                    self.mp_drawing.draw_landmarks(
                        image=frame, landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    self.mp_drawing.draw_landmarks(
                        image=frame, landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )

    def log_output(self, data):
        print(json.dumps(data, indent=2))
        with open(config.LOG_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return

        print("[INFO] Mirra Active.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                current_time = time.time()
                
                # Perform YOLOv8 Face Detection (30fps)
                detection_results = []
                if self.face_detector:
                    yolo_results = self.face_detector(frame, verbose=False, conf=config.CONFIDENCE_THRESHOLD)
                    for r in yolo_results:
                        if r.boxes:
                            # Extract [x1, y1, x2, y2, conf, cls]
                            all_detections = r.boxes.data.tolist()
                            if all_detections:
                                # Fix "Multiple People" issue: Sort by box area and pick the largest one
                                # Box area = (x2-x1) * (y2-y1)
                                all_detections.sort(key=lambda x: (x[2]-x[0]) * (x[3]-x[1]), reverse=True)
                                detection_results = [all_detections[0]] # Only focus on the primary user

                # Update smooth visualization every frame (30fps) using YOLO boxes
                self.visualize_frame(frame, detection_results)
                
                # Capture mesh results for validation
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mesh_results = self.face_mesh.process(frame_rgb)

                # Heavy AI Analysis every 2 seconds (ONLY if YOLO + Mesh agree)
                if current_time - self.last_detection_time >= config.DETECTION_INTERVAL:
                    result = self.process_frame(frame, detection_results, mesh_results)
                    self.log_output(result)
                    self.last_detection_time = current_time

                cv2.imshow('Mirra - Main Output', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    module = MirraModule()
    module.run()
