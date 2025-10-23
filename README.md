##Gesture Photo Booth## 
Author: Dalal Arafeh 
Python|OpenCV|NumPy|cvzone(MediaPipe Hands) 

This project is an interactive comuter vision system that transforms a regular webcam into a gesture-controlled photo booth. Replacing traditional input methods, it allows users to capture photos, apply filters, and create collages through natural gestures alone. 

The system begins by initializing the webcam using OpenCV and setting the capture resolution to 1280×720 for clarity and performance balance. Each incoming frame is mirrored using cv2.flip(frame, 1) to ensure natural alignment with the user’s movements, replicating a mirror effect. A dedicated save directory named booth is created to store all captured and processed images.

1. Frame acquisition and Initialization
   The core of gesture control relies on the HandDetector class from cvzone, which builds upon Google’s MediaPipe Hands model. The detector identifies up to 21 landmarks per hand, each corresponding to critical knuckle and fingertip coordinates in 3D space. These points are used to measure distances and infer hand posture. For this project, the hand detector operates at a moderate confidence threshold (detectionCon=0.35). The program continuously captures hand landmark data from the video stream, overlays skeletal outlines, and outputs a list of landmark coordinates (lmList) for gesture computation.

2. Hand Detection and Landmark Recognition
   The gesture control relies on the HandDetector class from cvzone, which builds upon Google's MediaPipe Hands mondel. It identifies up to 21 landmarks per hand, corresponding to knickle and fingertip coordinates in 3D space to measure distances and infer hand posture. For this project, the hand detector operates at (detectionCon=0.35). I started with 0.8, but the model was rejecting many frames because the detection confidence threshold was too strict.

3. Gesture Recognition and Mapping
   




















https://www.researchgate.net/publication/360732602_Towards_controlling_mouse_through_Hand_Gestures_A_novel_and_efficient_approach/fulltext/636b0a842f4bca7fd044ffcd/Towards-controlling-mouse-through-Hand-Gestures-A-novel-and-efficient-approach.pdf?origin=scientificContributions&__cf_chl_tk=OwwYDwP3RMETqo4oqdHQCTiWu_UGzBu.j2UpUvibEBE-1758978702-1.0.1.1-Y2PV4rlgBy8tMZKaJNPqL1n.ZTF8pgbtZ.ZcjlnPsPc 
