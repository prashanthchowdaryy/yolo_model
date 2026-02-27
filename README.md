🧠 YOLO-Based Real-Time Human Detection & Action Tracking
📌 Overview

This project is an end-to-end Computer Vision system built using YOLOv8, OpenCV, and Python for real-time human detection and tracking from video input.

The system detects all persons in a frame, selects the primary subject, and performs advanced visual tracking on the detected individual.

It demonstrates practical implementation of object detection pipelines used in real-world AI systems such as surveillance, sports analytics, and human-computer interaction.

🚀 Features

✅ Real-time person detection using YOLOv8

✅ Multi-person detection

✅ Main subject selection (largest bounding box logic)

✅ Background person counting

✅ Bounding box visualization

✅ Smooth video playback control

✅ Optimized inference pipeline

🛠 Tech Stack

Python 3.10

YOLOv8 (Ultralytics)

OpenCV

NumPy

MediaPipe (for extended pose integration)

📂 Project Structure
mediapipe_project/
│
├── real_time_body_detection.py
├── yolov8n.pt
├── input_video.mp4
├── README.md
└── requirements.txt
⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Create virtual environment (Python 3.10 recommended)
py -3.10 -m venv venv
venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt

Or manually:

pip install ultralytics opencv-python numpy mediapipe
▶️ Usage

Place your video file in the project directory.

Update the video path inside the script:

input_video = r"path_to_your_video.mp4"

Run:

python real_time_body_detection.py

Press Q to exit the window.

🧩 How It Works
🔍 Step 1: YOLO Detection

YOLOv8 detects all objects in each frame.

👤 Step 2: Person Filtering

Only detections labeled as "person" are selected.

🎯 Step 3: Main Subject Selection

The largest bounding box (by area) is chosen as the primary subject.

📊 Step 4: Counting

All detected persons are counted and displayed on screen.

🎥 Step 5: Visualization

Bounding boxes are drawn:

Red → Main subject

Green → Background persons

📈 Model Details

Model: yolov8n.pt

Architecture: YOLOv8 Nano

Input Resolution: 640x640 (default)

Real-time capable on CPU

🧠 Key Learnings

Understanding real-time inference pipelines

Optimizing detection frequency for performance

Handling multi-object detection

Confidence tuning

Frame-by-frame processing in OpenCV

Practical implementation of deep learning models

💡 Future Improvements

Add object tracking (DeepSORT / ByteTrack)

Save processed output video

Deploy as a Streamlit Web App

Convert into Android app

Add GPU acceleration

📌 Applications

Smart surveillance systems

Sports movement tracking

Human behavior analysis

AI-powered interaction systems

Security automation

🏁 Conclusion

This project strengthened practical understanding of object detection systems and real-time AI workflows.

Building and deploying YOLO-based pipelines bridges the gap between theory and production-level computer vision systems.
