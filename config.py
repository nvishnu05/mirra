import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
FACE_CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
AGE_PROTO = os.path.join(MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODELS_DIR, "gender_net.caffemodel")

# Output Settings
DEBUG_MODE = True # Set to True to see diagnostic metrics in logs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "mirra_logs.jsonl")

# Detection Settings
DETECTION_INTERVAL = 2.0  # seconds
CONFIDENCE_THRESHOLD = 0.50
HIGH_CONFIDENCE_THRESHOLD = 0.85
MIN_DETECTION_CONFIDENCE = 0.5
SMOOTHING_WINDOW_SIZE = 10  # Balanced for dynamic switching
EMOTION_SMOOTHING_WINDOW = 3 # More responsive for fleeting expressions

# Selection Weights
CENTER_WEIGHT = 0.6
SIZE_WEIGHT = 0.4

# Age Groups Mapping
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# PRD Groups: child (<12), teen (13–19), young_adult (20–35), middle_age (36–55), senior (56+)
AGE_MAP = {
    '(0-2)': 'child',
    '(4-6)': 'child',
    '(8-12)': 'child',
    '(15-20)': 'teen',
    '(25-32)': 'young_adult',
    '(38-43)': 'middle_age',
    '(48-53)': 'middle_age',
    '(60-100)': 'senior'
}

GENDER_LIST = ['male', 'female']

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
