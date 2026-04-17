import cv2
import numpy as np
import time
import json
import os
import config
import mediapipe as mp
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
        # MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE
        )

        # Age/Gender Models
        self.age_net = cv2.dnn.readNet(config.AGE_MODEL, config.AGE_PROTO)
        self.gender_net = cv2.dnn.readNet(config.GENDER_MODEL, config.GENDER_PROTO)
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

        # Image Enhancers
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # Buffers
        self.age_buffer = WeightedResultsBuffer()
        self.gender_buffer = WeightedResultsBuffer()
        self.emotion_buffer = WeightedResultsBuffer(size=config.EMOTION_SMOOTHING_WINDOW)

        self.last_detection_time = 0
        self.last_seen_time = time.time() # Track for buffer resetting


    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        if not results.detections:
            # If no person seen for > 5s, clear buffers to avoid ghosting attributes
            if time.time() - self.last_seen_time > 5.0:
                self.age_buffer.clear()
                self.gender_buffer.clear()
                self.emotion_buffer.clear()
            return {"timestamp": time.time(), "person_detected": False}

        self.last_seen_time = time.time() # Update timer as person is present

        frame_h, frame_w = frame.shape[:2]
        detection = results.detections[0] # Focus on primary detection
        bbox = detection.location_data.relative_bounding_box
        
        # 1. FACE ROI (for Gender/Age) - Increased padding to 30% for expression features
        fx, fy, fw, fh = int(bbox.xmin * frame_w), int(bbox.ymin * frame_h), int(bbox.width * frame_w), int(bbox.height * frame_h)
        fx1, fy1 = max(0, fx - int(fw * 0.3)), max(0, fy - int(fh * 0.3))
        fx2, fy2 = min(frame_w, fx + fw + int(fw * 0.3)), min(frame_h, fy + fh + int(fh * 0.3))
        face_roi = frame[fy1:fy2, fx1:fx2]


        # Demographics Processing
        if face_roi.size > 0:
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            
            # Gender
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            g_idx = gender_preds[0].argmax()
            gender_raw = config.GENDER_LIST[g_idx]
            gender_conf = float(gender_preds[0][g_idx])
            
            # Age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            a_idx = age_preds[0].argmax()
            age_raw = config.AGE_LIST[a_idx]
            age_val = config.AGE_MAP.get(age_raw, "unclear")
            age_conf = float(age_preds[0][a_idx])
            
            # Emotion (using DeepFace)
            try:
                # Use standard DeepFace analysis on face ROI
                emotion_results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                if emotion_results:
                    res = emotion_results[0]
                    emotion_raw = res['dominant_emotion']
                    emotion_val = config.EMOTION_MAP.get(emotion_raw, emotion_raw)
                    emotion_conf = float(res['emotion'][emotion_raw]) / 100.0 # DeepFace returns 0-100
                    
                    # Extract full proportions
                    emotion_proportions = {
                        config.EMOTION_MAP.get(k, k): round(float(v) / 100.0, 4) 
                        for k, v in res['emotion'].items()
                    }
                else:
                    emotion_val, emotion_conf, emotion_proportions = "neutral", 0.0, {}
            except Exception as e:
                print(f"[DEBUG] Emotion error: {e}")
                emotion_val, emotion_conf, emotion_proportions = "neutral", 0.0, {}
            
            # Buffer updates
            self.age_buffer.add(age_val, age_conf)
            self.gender_buffer.add(gender_raw, gender_conf)
            # Note: We buffer only the dominant emotion for stability
            self.emotion_buffer.add(emotion_val, emotion_conf)
            
            stable_age = self.age_buffer.get_stable_value(age_val)
            stable_gender = self.gender_buffer.get_stable_value(gender_raw)
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
                "people_in_frame": len(results.detections)
            }
            
            if config.DEBUG_MODE:
                result["debug"] = {
                    "status": "visual_roi_preview_active"
                }

            return result
        
        return {"timestamp": time.time(), "person_detected": False}

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
                if current_time - self.last_detection_time >= config.DETECTION_INTERVAL:
                    result = self.process_frame(frame)
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
