# Mirra — Smart Entrance Detection Module

Mirra is a real-time computer vision system designed to detect people entering a store and extract key attributes (gender, age, clothing style) to drive dynamic content.

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Models**:
   The system requires pre-trained Caffe models and OpenCV Haar Cascades. Run the downloader once:
   ```bash
   python models/downloader.py
   ```

3. **Run Detection**:
   ```bash
   python mirra_detector.py
   ```

## Key Features

- **2–3s Sampling**: Processes frames every 2 seconds to optimize performance.
- **Multi-Person Selection**: Uses proximity to center and bounding box size to select the primary person.
- **Demographic Extraction**: Predicts Age and Gender using Caffe models.
- **Clothing Style (Phase 1)**: Heuristic-based classification based on silhouette and edge density.
- **Privacy First**: All processing is local. No images or biometric data are stored externally.

## Output Structure

Detection results are logged in real-time to the terminal and appended to `mirra_logs.jsonl` in JSON format:

```json
{
  "timestamp": 1713000000.0,
  "person_detected": true,
  "gender": { "value": "male", "confidence": 0.91 },
  "age_group": { "value": "young_adult", "confidence": 0.85 },
  "clothing_style": { "value": "western", "confidence": 0.75 },
  "people_in_frame": 2,
  "selection_rule": "center_alignment"
}
```
