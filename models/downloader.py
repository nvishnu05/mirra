import os
import requests

MODELS = {
    "face_cascade": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        "filename": "haarcascade_frontalface_default.xml"
    },
    "age_proto": {
        "url": "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_deploy.prototxt",
        "filename": "age_deploy.prototxt"
    },
    "age_model": {
        "url": "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel",
        "filename": "age_net.caffemodel"
    },
    "gender_proto": {
        "url": "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt",
        "filename": "gender_deploy.prototxt"
    },
    "gender_model": {
        "url": "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel",
        "filename": "gender_net.caffemodel"
    },
}


def download_models(target_dir="models"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    for key, info in MODELS.items():
        filepath = os.path.join(target_dir, info["filename"])
        if not os.path.exists(filepath):
            print(f"Downloading {info['filename']}...")
            try:
                response = requests.get(info["url"], stream=True, timeout=30)
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully downloaded {info['filename']}")
            except Exception as e:
                print(f"Error downloading {info['filename']}: {e}")
        else:
            print(f"{info['filename']} already exists.")

if __name__ == "__main__":
    download_models()
