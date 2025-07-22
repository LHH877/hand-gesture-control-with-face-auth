# hand-gesture-control-with-face-auth
A smart control system using MediaPipe-based hand gesture recognition secured with facial recognition and integrated with Firebase for real-time user data management.

# Hand Gesture Control System Secured by Facial Recognition

This project combines real-time hand gesture recognition with facial authentication and a Firebase-integrated backend for smart control systems.

## 🔧 Features
- Hand gesture recognition using **MediaPipe**
- Facial recognition using **OpenCV + LBPH**
- Real-time data storage and authentication via **Firebase**
- Rule-based algorithm for gesture classification


### Facial Recognition with Role Detection
<img src="hand-gesture-control-with-face-auth/ImagesForDemo/image1.png" width="400"/>

Facial recognition identifies registered users and displays their name and department.

---

### Real-Time Hand Gesture Recognition
<img src="hand-gesture-control-with-face-auth/ImagesForDemo/image2.png" width="400"/>

The system detects finger gestures using MediaPipe and overlays keypoints in real time.

---

### Realtime Firebase Activity Logging
<img src="hand-gesture-control-with-face-auth/ImagesForDemo/image3.png" width="400"/>

Gesture changes are logged in Firebase with timestamps and user identity.


## 🛠️ Technologies Used
- Python
- MediaPipe
- OpenCV (LBPH)
- Firebase (Realtime Database)
- Pyrebase

## 📁 Folder Structure
- `gesture_module/` – MediaPipe logic
- `face_auth/` – Face recognition authentication
- `firebase/` – Firebase database logic
- `main.py` – Entry point

## 🚀 Setup

1. Clone the repo:
```bash
git clone https://github.com/yourusername/hand-gesture-control-with-face-auth.git
