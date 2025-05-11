# File: main_app.py

import cv2 as cv
import numpy as np
import joblib
from hand_tracker import HandTracker # Ensure this is correctly defined
import pyttsx3
import time
import mediapipe as mp
import json # For loading labels map

# --- Configuration ---
LABELS_JSON_FILE = "labels.json"
MODEL_PATH = "multisign_model.pkl" # Path to your new multi-sign model

# These should match the settings used during preprocessing for the loaded model
EXPECTED_NUM_FEATURES = 84 # e.g., 42 for 1 hand, 84 for 2 hands
MAX_HANDS_FOR_MODEL = 2    # 1 or 2, matching the model

# Hand Tracking and Model Confidence
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7
PREDICTION_CONFIDENCE_THRESHOLD = 0.75 # Adjusted from 0.80, tune as needed
STABILITY_THRESHOLD_FRAMES = 10

# Word Formation Logic
WORD_COMPLETION_TIMEOUT = 2.0
current_word = []
last_activity_time = time.time()

# --- Load Labels Map ---
def load_labels_map_inv(json_path):
    try:
        with open(json_path, 'r') as f:
            labels_map = json.load(f)
        labels_map_inv = {v: k for k, v in labels_map.items()} # int_label: str_label
        print(f"Loaded inverse labels map for live prediction: {labels_map_inv}")
        return labels_map_inv
    except Exception as e:
        print(f"Error loading inverse labels map from {json_path}: {e}")
        return None

labels_map_inv = load_labels_map_inv(LABELS_JSON_FILE)
if not labels_map_inv:
    print("Exiting due to missing labels map.")
    exit()

# --- Initialize HandTracker, Model, TTS (Same as your last version) ---
hand_tracker = HandTracker(max_num_hands=MAX_HANDS_FOR_MODEL,
                           min_detection_confidence=DETECTION_CONFIDENCE,
                           min_tracking_confidence=TRACKING_CONFIDENCE)
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model '{MODEL_PATH}': {e}")
    exit()
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    tts_engine = None

# --- Normalization & Feature Extraction (Same as your last version, ensure consistency) ---
def normalize_single_hand_landmarks_live(hand_landmarks_object):
    # ... (ensure this function is identical to the one used in preprocess_data.py) ...
    # ... (and matches the FEATURES_PER_HAND calculation, e.g. 42 features) ...
    if not hand_landmarks_object or not hand_landmarks_object.landmark: return None
    landmarks = hand_landmarks_object.landmark
    if len(landmarks) != 21: return None # TARGET_NUM_LANDMARKS_PER_HAND
    normalized_coords = []
    try:
        wrist_x, wrist_y = landmarks[0].x, landmarks[0].y
        mcp_middle_x, mcp_middle_y = landmarks[9].x, landmarks[9].y
        scale_distance = np.sqrt((mcp_middle_x - wrist_x)**2 + (mcp_middle_y - wrist_y)**2)
        if scale_distance < 1e-6: return None
        for lm_point in landmarks:
            normalized_coords.extend([(lm_point.x - wrist_x) / scale_distance, (lm_point.y - wrist_y) / scale_distance])
        return normalized_coords if len(normalized_coords) == (EXPECTED_NUM_FEATURES // MAX_HANDS_FOR_MODEL if MAX_HANDS_FOR_MODEL > 0 else EXPECTED_NUM_FEATURES) else None # Should be 42
    except Exception: return None


def get_feature_vector_live(detection_results):
    # ... (ensure this function is identical to how features were created for the loaded model) ...
    # ... (respecting MAX_HANDS_FOR_MODEL and resulting in EXPECTED_NUM_FEATURES) ...
    features_per_hand = EXPECTED_NUM_FEATURES // MAX_HANDS_FOR_MODEL if MAX_HANDS_FOR_MODEL > 0 else EXPECTED_NUM_FEATURES

    if MAX_HANDS_FOR_MODEL == 1:
        if detection_results.multi_hand_landmarks:
            hand_obj = hand_tracker.get_landmarks(detection_results, hand_index=0)
            return normalize_single_hand_landmarks_live(hand_obj)
        return None
    elif MAX_HANDS_FOR_MODEL == 2:
        combined_features = [0.0] * EXPECTED_NUM_FEATURES # Initialize with zeros
        if detection_results.multi_hand_landmarks:
            num_detected = len(detection_results.multi_hand_landmarks)
            processed_left, processed_right = False, False
            for i in range(min(num_detected, MAX_HANDS_FOR_MODEL)):
                hand_obj = detection_results.multi_hand_landmarks[i]
                norm_hand = normalize_single_hand_landmarks_live(hand_obj)
                if norm_hand:
                    handedness = "Unknown"
                    if detection_results.multi_handedness and i < len(detection_results.multi_handedness):
                        classification_list = detection_results.multi_handedness[i]
                        if classification_list.classification:
                            handedness = classification_list.classification[0].label
                    
                    if handedness == 'Left' and not processed_left:
                        combined_features[:features_per_hand] = norm_hand
                        processed_left = True
                    elif handedness == 'Right' and not processed_right:
                        combined_features[features_per_hand:] = norm_hand
                        processed_right = True
                    elif not processed_left : # Fallback: fill first available slot
                        combined_features[:features_per_hand] = norm_hand
                        processed_left = True
                    elif not processed_right:
                        combined_features[features_per_hand:] = norm_hand
                        processed_right = True
            if processed_left or processed_right: return combined_features
        return None
    return None

# --- Speak Function (Same as your last version) ---
def speak_text(text_to_speak):
    if tts_engine and text_to_speak:
        try:
            print(f"Saying: {text_to_speak}")
            tts_engine.say(text_to_speak)
            tts_engine.runAndWait()
        except Exception as e: print(f"TTS Error: {e}")

# --- Main Application Loop (largely same, but uses loaded labels_map_inv) ---
def main():
    global current_word, last_activity_time
    if not labels_map_inv: # Check if map loaded
        print("Cannot start: Labels map not loaded.")
        return

    cap = cv.VideoCapture(0)
    # ... (rest of main loop from your previous main_app.py) ...
    # ... ensure current_char_recognized_this_frame uses labels_map_inv ...
    # ... and logic for "Unknown" if confidence is low ...

    print("Starting live sign to word... Press 'q' to quit. Press SPACE to clear word.")
    last_predicted_char_for_stability = None
    stable_prediction_counter = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame_for_display = cv.flip(frame.copy(), 1)
        frame_rgb_for_processing = cv.flip(cv.cvtColor(frame, cv.COLOR_BGR2RGB), 1)

        frame_rgb_for_processing.flags.writeable = False
        detection_results = hand_tracker.process_frame(frame_rgb_for_processing)
        frame_rgb_for_processing.flags.writeable = True

        current_char_recognized_this_frame = "" 
        hand_is_detected_this_frame = bool(detection_results.multi_hand_landmarks)

        if hand_is_detected_this_frame:
            frame_for_display = hand_tracker.draw_landmarks(frame_for_display, detection_results)
            normalized_features = get_feature_vector_live(detection_results)

            if normalized_features:
                feature_vector = np.array([normalized_features])
                if feature_vector.shape[1] == EXPECTED_NUM_FEATURES:
                    prediction_proba = model.predict_proba(feature_vector)[0]
                    predicted_numerical_label = np.argmax(prediction_proba)
                    confidence = prediction_proba[predicted_numerical_label]

                    if confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
                        current_char_recognized_this_frame = labels_map_inv.get(predicted_numerical_label, "ERR") 
                    else:
                        current_char_recognized_this_frame = "Unknown"
                    last_activity_time = time.time()
        
        if current_char_recognized_this_frame:
            if current_char_recognized_this_frame == last_predicted_char_for_stability:
                stable_prediction_counter += 1
            else:
                last_predicted_char_for_stability = current_char_recognized_this_frame
                stable_prediction_counter = 1
            
            if stable_prediction_counter == STABILITY_THRESHOLD_FRAMES:
                stable_recognized_char = last_predicted_char_for_stability
                if stable_recognized_char != "Unknown" and stable_recognized_char != "ERR":
                    if not current_word or current_word[-1] != stable_recognized_char:
                        current_word.append(stable_recognized_char)
                        print(f"Letter added: {stable_recognized_char}. Current word: {''.join(current_word)}")
                elif stable_recognized_char == "Unknown":
                    if not current_word: speak_text("Unknown")
                    print("Sign recognized as: Unknown")
                last_activity_time = time.time()
                stable_prediction_counter = 0 
                last_predicted_char_for_stability = None 
        
        elif hand_is_detected_this_frame and not current_char_recognized_this_frame:
            stable_prediction_counter = 0
            last_predicted_char_for_stability = None
        
        if current_word:
            time_since_last_activity = time.time() - last_activity_time
            speak_condition_met = False
            if not hand_is_detected_this_frame and time_since_last_activity > WORD_COMPLETION_TIMEOUT:
                speak_condition_met = True
            elif hand_is_detected_this_frame and \
                 (not current_char_recognized_this_frame or current_char_recognized_this_frame == "Unknown") and \
                 stable_prediction_counter < STABILITY_THRESHOLD_FRAMES and \
                 time_since_last_activity > (WORD_COMPLETION_TIMEOUT + 1.0):
                speak_condition_met = True

            if speak_condition_met:
                formed_word = "".join(current_word)
                speak_text(formed_word)
                current_word = []
                last_predicted_char_for_stability = None
                stable_prediction_counter = 0
                last_activity_time = time.time()

        display_text_on_screen = f"Word: {''.join(current_word)}"
        if current_char_recognized_this_frame and current_char_recognized_this_frame not in ["Unknown", "ERR"]:
            display_text_on_screen += f" ({current_char_recognized_this_frame}?)"
        elif current_char_recognized_this_frame == "Unknown":
             display_text_on_screen += f" (Unknown?)"
        elif last_predicted_char_for_stability and stable_prediction_counter > 0 :
            display_text_on_screen += f" ({last_predicted_char_for_stability}...{stable_prediction_counter})"

        cv.putText(frame_for_display, display_text_on_screen, (30, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow('Dhwani - Sign to Word', frame_for_display)

        key = cv.waitKey(5) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '):
            print("Word cleared by user.")
            current_word = []
            last_predicted_char_for_stability = None
            stable_prediction_counter = 0
            last_activity_time = time.time()

    hand_tracker.close()
    cap.release()
    cv.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    main()