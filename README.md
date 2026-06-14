# OmniMedia Deepfake Detection Platform

## Overview

OmniMedia Deepfake Detection Platform is a multimodal AI-powered forensic system designed to detect manipulated media content. Unlike traditional approaches that rely on a single model, OmniMedia employs a Late Fusion Architecture that combines visual and acoustic analysis to improve deepfake detection accuracy.

The platform can analyze both videos and audio files and generate an explainable forensic report highlighting potential signs of AI-generated manipulation.

---

## Features

* 🎥 Video Deepfake Detection using Vision Transformers (ViT)
* 🎙️ AI Voice Clone Detection using Acoustic Neural Networks
* 🧠 Late Fusion Meta-Classifier for final decision making
* 📊 Interactive forensic dashboard
* 🔥 Visual anomaly heatmaps
* 📈 Explainable AI-generated reports
* ⚡ FastAPI-powered web interface
* 📂 Support for MP4 and WAV files

---

## System Architecture

The platform consists of three major AI components:

### 1. Visual Analysis Module

* Uses a Vision Transformer (ViT)
* Analyzes complete video frames
* Detects diffusion artifacts and visual inconsistencies

### 2. Audio Analysis Module

* Extracts MFCC (Mel-Frequency Cepstral Coefficients)
* Processes 3-second audio segments
* Detects synthetic speech patterns and voice cloning artifacts

### 3. Fusion Module

* Combines visual and acoustic confidence scores
* Uses a trained Meta-Classifier
* Produces the final authenticity verdict

---

## Tech Stack

### Backend

* Python
* FastAPI
* Uvicorn

### Machine Learning

* PyTorch
* TorchVision
* Hugging Face Transformers
* Librosa
* MTCNN

### Data Processing

* OpenCV
* NumPy
* Pandas
* SoundFile

### Frontend

* HTML
* CSS
* JavaScript

---

## Project Structure

```text
Deepfake_Platform/
│
├── app/
│   ├── main.py
│   ├── services/
│   │   ├── video_service.py
│   │   ├── audio_service.py
│   │   ├── model_service.py
│   │   ├── fusion_service.py
│   │   └── explain_service.py
│
├── models/
│   ├── vit_deepfake_weights.pt
│   ├── audio_deepfake_weights.pt
│   └── fusion_weights.pt
│
├── generate_omfb_dataset.py
├── ingest_custom_video.py
├── train_vision.py
├── train_audio.py
├── train_fusion.py
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/OmniMedia-Deepfake-Detection.git
cd OmniMedia-Deepfake-Detection
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install fastapi uvicorn python-multipart torch torchvision torchaudio transformers librosa opencv-python mtcnn datasets soundfile pandas tqdm pillow
```

---

## Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Open your browser:

```text
http://127.0.0.1:8000
```

Upload:

* MP4 videos
* WAV audio files

The platform will automatically perform forensic analysis and generate a detection report.

---

## Model Pipeline

```text
Input Media
      │
      ▼
 ┌───────────┐
 │ Video AI  │
 └───────────┘
      │
      ▼
Visual Score
      │
      ├──────────────┐
      │              │
      ▼              ▼
Audio Analysis    Audio Score
      │              │
      └──────┬───────┘
             ▼
     Meta-Classifier
             ▼
     Final Verdict
```

---

## Future Enhancements

* Real-time deepfake detection
* Multi-language voice clone analysis
* Explainable AI visualizations
* Cloud deployment
* Mobile application support
* Deepfake localization within videos

---

## Disclaimer

This project is intended for educational, research, and cybersecurity purposes only. Detection results should be treated as probabilistic assessments and not as definitive legal evidence.

---

## Author

Jyotiradithya Reddy
Computer Science and Engineering
National Institute of Technology Delhi (NIT Delhi)
