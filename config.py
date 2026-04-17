import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# YOLO Models
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
YOLO_FACE_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n-face.pt")

# Output Settings
DEBUG_MODE = True # Set to True to see diagnostic metrics in logs
VISUALIZE_FACE_MESH = True # Draw landmarks on the main output
VISUALIZE_BOUNDING_BOX = True # Draw face tracking box
LOG_FILE = os.path.join(BASE_DIR, "mirra_logs.jsonl")

# Detection Settings
DETECTION_INTERVAL = 2.0  # seconds
CONFIDENCE_THRESHOLD = 0.70 # Increased to ignore distant objects/shirts
MIN_DETECTION_CONFIDENCE = 0.7 
SILENT_MODE = True # Suppress DeepFace log spam
SMOOTHING_WINDOW_SIZE = 10  # Balanced for dynamic switching
EMOTION_SMOOTHING_WINDOW = 3 # More responsive for fleeting expressions

# Selection Weights
CENTER_WEIGHT = 0.6
SIZE_WEIGHT = 0.4

# Age Groups Mapping (For DeepFace Numeric Output)
# Labels: child (<13), teen (13-19), young_adult (20-35), middle_age (36-55), senior (56+)
AGE_GROUPS = {
    "child": (0, 12),
    "teen": (13, 19),
    "young_adult": (20, 35),
    "middle_age": (36, 55),
    "senior": (56, 110)
}

GENDER_MAP = {
    'Man': 'male',
    'Woman': 'female'
}

# DeepFace Emotion Settings
EMOTION_ACTIONS = ['emotion']
EMOTION_MAP = {
    'happy': 'happy',
    'sad': 'sad',
    'neutral': 'neutral',
    'surprise': 'surprised',
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear'
}
