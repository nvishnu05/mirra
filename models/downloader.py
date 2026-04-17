import os
import requests

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Success!")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
    
    YOLO_URLS = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "yolov8n-face.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-face.pt"
    }
    
    for filename, url in YOLO_URLS.items():
        target = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(target):
            download_file(url, target)
        else:
            print(f"{filename} already exists in models/ directory.")
