# File: README.md

# Dhwani - Sign Language to Speech

Dhwani is a Python-based project aimed at bridging the communication gap for the Deaf and Hard of Hearing community by translating sign language (currently focusing on individual letters and simple words like "hello") into audible English speech.

## Project Goal

The primary objective is to develop a system that can:
1.  Capture hand gestures using a webcam.
2.  Process these gestures using computer vision and machine learning techniques.
3.  Recognize predefined signs (letters and words).
4.  Convert the recognized sequence of signs into text.
5.  Speak out the resulting text using a Text-to-Speech (TTS) engine.

## Key Technologies Used

* **Python 3.x**
* **OpenCV (`cv2`)**: For camera input and basic image/video processing.
* **MediaPipe**: For real-time hand tracking and landmark detection.
* **NumPy**: For numerical operations and handling feature data.
* **Scikit-learn (`sklearn`)**: For training machine learning models (e.g., SVM, RandomForest).
* **Joblib**: For saving and loading trained models.
* **Matplotlib & Seaborn**: For data visualization (e.g., confusion matrix).
* **pyttsx3**: For offline Text-to-Speech functionality.
* **JSON**: For managing sign labels.

## Project Structure (Example)