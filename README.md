# Hand Gesture Recognition

A **real-time hand gesture recognition system** built using **MediaPipe**, **OpenCV**, and **Machine Learning**.  
This project detects and classifies hand gestures like `open_palm`, `index_up`, `thumbs_up`, and `peace` using a webcam feed.

---

## Features

-  Real-time hand tracking using **MediaPipe**
-  Gesture classification with a trained **ML model (`gesture_model.pkl`)**
-  Supports multiple gesture types
-  Easily extendable for gesture-based control (e.g., media playback, mouse movement, volume control)

---

##  Tech Stack

| Tool           | Purpose                                      |
|----------------|----------------------------------------------|
| Python 3.x     | Core programming language                    |
| OpenCV         | Video capture and image processing           |
| MediaPipe      | Hand landmark detection                      |
| Scikit-learn   | Machine learning model for gesture detection |
| PyAutoGUI      | System automation based on gestures          |

---

##  Installation

### 1️ Clone the repository

```bash
git clone https://github.com/Pallavi-Challagondla/HandGestureRecognition.git
cd HandGestureRecognition

```

### 2 Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate

```


### 3 Install dependencies
```bash
pip install opencv-python mediapipe scikit-learn pyautogui
```

### 4 Run the project
```bash
python hand_tracking.py
```

