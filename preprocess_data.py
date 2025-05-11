# File: preprocess_data.py
import cv2 as cv
import os
import numpy as np
import json # For loading the labels map
from hand_tracker import HandTracker # Your HandTracker class
import mediapipe as mp # Should be imported in hand_tracker.py, but good here too if used directly

# --- Configuration ---
VIDEO_DATA_DIR = "videos"  # Main directory containing subfolders for each label
LABELS_JSON_FILE = "labels.json" # Path to your new JSON label map
OUTPUT_DATA_X_FILE = "X_multisign_data.npy" # New name for the output file
OUTPUT_DATA_Y_FILE = "y_multisign_data.npy" # New name for the output file

MAX_FRAMES_PER_VIDEO = 30  # Process up to this many frames per video, or None to process all
TARGET_NUM_LANDMARKS_PER_HAND = 21
NUM_COORDS_PER_LANDMARK = 2 # Using x, y
FEATURES_PER_HAND = TARGET_NUM_LANDMARKS_PER_HAND * NUM_COORDS_PER_LANDMARK # 42
MAX_HANDS_TO_PROCESS = 2 # Process up to two hands
TOTAL_FEATURES_EXPECTED = FEATURES_PER_HAND * MAX_HANDS_TO_PROCESS # e.g., 84

# Initialize HandTracker
hand_tracker = HandTracker(max_num_hands=MAX_HANDS_TO_PROCESS,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)

def load_labels_map(json_path):
    try:
        with open(json_path, 'r') as f:
            labels_map = json.load(f)
        print(f"Loaded labels map from {json_path}: {labels_map}")
        return labels_map
    except FileNotFoundError:
        print(f"Error: Labels JSON file not found at {json_path}. Please create it.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}. Please check its format.")
        return None

def normalize_single_hand_landmarks(hand_landmarks_object):
    if not hand_landmarks_object or not hand_landmarks_object.landmark: return None
    landmarks = hand_landmarks_object.landmark
    if len(landmarks) != TARGET_NUM_LANDMARKS_PER_HAND: return None
    
    normalized_coords = []
    try:
        wrist_x, wrist_y = landmarks[0].x, landmarks[0].y
        mcp_middle_x, mcp_middle_y = landmarks[9].x, landmarks[9].y
        scale_distance = np.sqrt((mcp_middle_x - wrist_x)**2 + (mcp_middle_y - wrist_y)**2)
        if scale_distance < 1e-6: return None

        for lm_point in landmarks:
            norm_x = (lm_point.x - wrist_x) / scale_distance
            norm_y = (lm_point.y - wrist_y) / scale_distance
            normalized_coords.extend([norm_x, norm_y])
        
        return normalized_coords if len(normalized_coords) == FEATURES_PER_HAND else None
    except IndexError: return None
    except Exception as e:
        print(f"Error in normalize_single_hand_landmarks: {e}")
        return None

def extract_features_from_video(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    video_frame_features_list = []
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        if MAX_FRAMES_PER_VIDEO is not None and frame_count >= MAX_FRAMES_PER_VIDEO: break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        detection_results = hand_tracker.process_frame(frame_rgb)
        frame_rgb.flags.writeable = True

        # Initialize feature vector for this frame (e.g., 84 features for 2 hands)
        current_frame_feature_vector = [0.0] * TOTAL_FEATURES_EXPECTED
        processed_at_least_one_hand = False

        if detection_results.multi_hand_landmarks:
            num_detected = len(detection_results.multi_hand_landmarks)
            
            # Prioritize Left then Right for consistent ordering if possible
            processed_left_features = None
            processed_right_features = None

            for i in range(min(num_detected, MAX_HANDS_TO_PROCESS)):
                hand_obj = detection_results.multi_hand_landmarks[i]
                norm_hand = normalize_single_hand_landmarks(hand_obj)
                if norm_hand:
                    handedness = "Unknown" # Default if no handedness info
                    if detection_results.multi_handedness and i < len(detection_results.multi_handedness):
                        classification_list = detection_results.multi_handedness[i]
                        if classification_list.classification:
                            handedness = classification_list.classification[0].label

                    if handedness == 'Left' and processed_left_features is None:
                        processed_left_features = norm_hand
                    elif handedness == 'Right' and processed_right_features is None:
                        processed_right_features = norm_hand
                    elif handedness == "Unknown": # Fallback for unknown or if one slot is already taken
                        if processed_left_features is None: processed_left_features = norm_hand
                        elif processed_right_features is None: processed_right_features = norm_hand


            if processed_left_features:
                current_frame_feature_vector[:FEATURES_PER_HAND] = processed_left_features
                processed_at_least_one_hand = True
            if processed_right_features:
                current_frame_feature_vector[FEATURES_PER_HAND:TOTAL_FEATURES_EXPECTED] = processed_right_features
                processed_at_least_one_hand = True
        
        if processed_at_least_one_hand: # Only add if we got some valid hand data
            video_frame_features_list.append(current_frame_feature_vector)
        frame_count += 1
        
    cap.release()
    return video_frame_features_list

def main():
    labels_map = load_labels_map(LABELS_JSON_FILE)
    if not labels_map:
        return

    all_features = []
    all_labels = []

    if not os.path.isdir(VIDEO_DATA_DIR):
        print(f"Error: Video data directory '{VIDEO_DATA_DIR}' not found.")
        return

    # Iterate based on the labels defined in the JSON file
    for sign_name, numerical_label in labels_map.items():
        sign_video_folder = os.path.join(VIDEO_DATA_DIR, sign_name)
        if not os.path.isdir(sign_video_folder):
            print(f"Warning: Directory for sign '{sign_name}' not found at '{sign_video_folder}'")
            continue

        print(f"Processing videos for sign: '{sign_name}' (Label: {numerical_label})...")
        for video_filename in os.listdir(sign_video_folder):
            if video_filename.lower().endswith((".mp4", ".mov", ".avi")):
                video_path = os.path.join(sign_video_folder, video_filename)
                print(f"  Processing video: {video_filename}")
                
                video_frame_features = extract_features_from_video(video_path)
                
                if video_frame_features:
                    all_features.extend(video_frame_features)
                    all_labels.extend([numerical_label] * len(video_frame_features))
                    print(f"    Extracted {len(video_frame_features)} feature sets from {video_filename}")
                else:
                    print(f"    No valid features extracted from {video_filename}")

    if not all_features:
        print("No features were extracted. Please check video paths and content, and ensure labels.json is configured.")
        hand_tracker.close()
        return

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\n--- Data Preprocessing Complete ---")
    print(f"Total feature sets (samples): {X.shape[0]}")
    if X.shape[0] > 0:
        print(f"Number of features per sample: {X.shape[1]}") # Should be TOTAL_FEATURES_EXPECTED
    else:
        print("Number of features per sample: N/A (no samples)")
    print(f"Number of labels: {y.shape[0]}")
    print(f"Unique numerical labels found in processed data: {np.unique(y)}")

    np.save(OUTPUT_DATA_X_FILE, X)
    np.save(OUTPUT_DATA_Y_FILE, y)
    print(f"Feature data saved to: {OUTPUT_DATA_X_FILE}")
    print(f"Label data saved to: {OUTPUT_DATA_Y_FILE}")

    hand_tracker.close()

if __name__ == '__main__':
    main()