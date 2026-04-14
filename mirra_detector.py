import cv2
import numpy as np
import time
import json
import os
import config
import mediapipe as mp
from collections import Counter

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
        self.clothing_buffer = WeightedResultsBuffer()

        self.last_detection_time = 0
        self.last_seen_time = time.time() # Track for buffer resetting
        self.last_clothing_style = "unclear" # For hysteresis

    def get_clothing_style(self, torso_roi, last_style="unclear"):
        """
        Structural clothing analysis using provided torso region.
        Applies a strict 0.60 confidence gate.
        """
        if torso_roi is None or torso_roi.size == 0:
            return "unclear", 0.0

        gray = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1. Structural Checks (Placket & Corners)
        h, w = gray.shape
        center_zone = gray[:, int(w*0.35):int(w*0.65)]
        sobel_x = cv2.Sobel(center_zone, cv2.CV_64F, 1, 0, ksize=3)
        placket_score = np.mean(np.abs(sobel_x)) / 255.0
        
        corners = cv2.goodFeaturesToTrack(gray, 25, config.CORNER_QUALITY, 10)
        corner_count = len(corners) if corners is not None else 0
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Hysteresis Thresholds (Lowered bar for currently active style)
        p_threshold = config.PLACKET_THRESHOLD
        t_complexity = config.TEXTURE_COMPLEXITY_THRESHOLD
        
        if last_style == "formal":
            p_threshold *= config.STABILITY_MARGIN
        elif last_style == "traditional":
            t_complexity *= config.STABILITY_MARGIN

        # Classification Heuristics
        is_traditional = False
        
        # Formal: Structural Placket + Corners (Strict: High structure, Low texture + Dominance)
        if (placket_score > p_threshold and 
            corner_count >= config.CORNER_COUNT_THRESHOLD and 
            edge_density < config.MAX_FORMAL_TEXTURE and
            placket_score > (edge_density * config.DOMINANCE_FACTOR)):
            value = "formal"
            confidence = 0.85
        
        # Traditional High Texture (Sarees/Silk patterns) 
        elif edge_density > t_complexity * 1.5:
            is_traditional = True
        
        # Traditional Low Structure (Kurtas/Simple ethnic)
        elif placket_score < config.MAX_FORMAL_STRUCTURE and edge_density > config.MIN_TRADITIONAL_DENSITY:
            is_traditional = True
        
        # Western: Lowered priority fallback (User Spec: 0.02)
        elif edge_density > 0.02:
            value = "western"
            confidence = 0.80
        else:
            value = "unclear"
            confidence = 0.40

        if is_traditional:
            value = "traditional"
            # PROPORTIONAL CONFIDENCE: Scaled based on edge density
            # Density 0.015 -> ~0.70 confidence, Density 0.20 -> ~0.90 confidence
            confidence = min(0.95, config.TRADITIONAL_BASE_CONFIDENCE + (edge_density * 1.2))

        # Reduced confidence gate (Let the buffer handle smoothing)
        if confidence < 0.40:
            return "unclear", confidence

        return value, confidence, (edge_density, placket_score)

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        if not results.detections:
            # If no person seen for > 5s, clear buffers to avoid ghosting attributes
            if time.time() - self.last_seen_time > 5.0:
                self.age_buffer.clear()
                self.gender_buffer.clear()
                self.clothing_buffer.clear()
                self.last_clothing_style = "unclear"
            return {"timestamp": time.time(), "person_detected": False}

        self.last_seen_time = time.time() # Update timer as person is present

        frame_h, frame_w = frame.shape[:2]
        detection = results.detections[0] # Focus on primary detection
        bbox = detection.location_data.relative_bounding_box
        
        # 1. FACE ROI (for Gender/Age) - Standard 20% padding
        fx, fy, fw, fh = int(bbox.xmin * frame_w), int(bbox.ymin * frame_h), int(bbox.width * frame_w), int(bbox.height * frame_h)
        pad_w, pad_h = int(fw * 0.2), int(fh * 0.2)
        fx1, fy1 = max(0, fx - pad_w), max(0, fy - pad_h)
        fx2, fy2 = min(frame_w, fx + fw + pad_w), min(frame_h, fy + fh + pad_h)
        face_roi = frame[fy1:fy2, fx1:fx2]

        # 2. TORSO ROI (for Clothing) - User Spec: Include neck/collar region
        tx1 = max(0, fx - fw)
        tx2 = min(frame_w, fx + 2*fw)
        
        # Start strictly below chin (fy + 1.1*fh) to avoid mouth interference
        ty1 = fy + int(fh * 1.1)
        ty2 = fy + int(fh * 6.0) # Extended downward to capture lower garments/patterns
        
        # Boundary Clamping
        tx1, ty1 = max(0, tx1), max(0, ty1)
        tx2, ty2 = min(frame_w, tx2), min(frame_h, ty2)
        
        clothing_val = "unclear"
        clothing_conf = 0.0
        e_density, p_score = 0.0, 0.0
        
        # VALIDATION
        if (ty2 - ty1) > fh * 0.5:
            torso_roi = frame[ty1:ty2, tx1:tx2]
            if torso_roi.size > 0:
                # NORMALIZE (User Spec: 224x224)
                torso_roi = cv2.resize(torso_roi, (224, 224))
                cv2.imshow("Mirra - Clothing Debug ROI", torso_roi)
                clothing_val, clothing_conf, metrics = self.get_clothing_style(torso_roi, self.last_clothing_style)
                e_density, p_score = metrics
        else:
            clothing_val = "unclear"

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
            
            # Buffer updates
            self.age_buffer.add(age_val, age_conf)
            self.gender_buffer.add(gender_raw, gender_conf)
            self.clothing_buffer.add(clothing_val, clothing_conf)
            
            stable_age = self.age_buffer.get_stable_value(age_val)
            stable_gender = self.gender_buffer.get_stable_value(gender_raw)
            stable_clothing = self.clothing_buffer.get_stable_value(clothing_val)
            self.last_clothing_style = stable_clothing # Update state for next frame's hysteresis

            result = {
                "timestamp": time.time(),
                "person_detected": True,
                "gender": {"value": stable_gender, "confidence": round(gender_conf, 2)},
                "age_group": {"value": stable_age, "confidence": round(age_conf, 2)},
                "clothing_style": {"value": stable_clothing, "confidence": round(clothing_conf, 2)},
                "people_in_frame": len(results.detections)
            }
            
            if config.DEBUG_MODE:
                result["debug"] = {
                    "edge_density": round(e_density, 3),
                    "placket_score": round(p_score, 3),
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

        print("[INFO] Mirra Active. Clothing Preview window is visible.")
        
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
