# File: hand_tracker.py

import cv2 as cv
import mediapipe as mp

class HandTracker:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands_module = mp.solutions.hands
        self.hands = self.mp_hands_module.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame_rgb):
        return self.hands.process(frame_rgb)

    def draw_landmarks(self, frame_bgr, detection_results):
        if detection_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks_object in enumerate(detection_results.multi_hand_landmarks):
                handedness_text = ""
                # Corrected to use hand_idx
                if detection_results.multi_handedness and hand_idx < len(detection_results.multi_handedness):
                    handedness_classification_list = detection_results.multi_handedness[hand_idx]
                    if handedness_classification_list.classification: # Check if classification list is not empty
                        handedness_text = handedness_classification_list.classification[0].label

                self.mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks_object,
                    self.mp_hands_module.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                if handedness_text: # Only draw if text is not empty
                    if hand_landmarks_object.landmark and frame_bgr.shape[0] > 0 and frame_bgr.shape[1] > 0:
                        try:
                            wrist_landmark = hand_landmarks_object.landmark[0]
                            text_x = int(wrist_landmark.x * frame_bgr.shape[1]) - 20
                            text_y = int(wrist_landmark.y * frame_bgr.shape[0]) - 10
                            # Ensure text coordinates are within frame boundaries for safety
                            text_x = max(10, min(text_x, frame_bgr.shape[1] - 10))
                            text_y = max(10, min(text_y, frame_bgr.shape[0] - 10))
                            cv.putText(frame_bgr, handedness_text, (text_x, text_y),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv.LINE_AA)
                        except IndexError:
                             pass # In case landmark[0] is somehow not available

        return frame_bgr

    def get_landmarks(self, detection_results, hand_index=0):
        if detection_results.multi_hand_landmarks:
            if hand_index < len(detection_results.multi_hand_landmarks):
                return detection_results.multi_hand_landmarks[hand_index]
        return None

    def close(self):
        self.hands.close()

# Example Usage (for testing this module directly)
if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.7) # Test with 2 hands

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_flipped = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame_flipped, cv.COLOR_BGR2RGB)

        frame_rgb.flags.writeable = False
        results = tracker.process_frame(frame_rgb)
        frame_rgb.flags.writeable = True

        # Draw on the BGR flipped frame
        annotated_image = tracker.draw_landmarks(frame_flipped.copy(), results) 

        # Test get_landmarks (optional)
        # if results.multi_hand_landmarks:
        #     for i in range(len(results.multi_hand_landmarks)):
        #         hand_lm_data = tracker.get_landmarks(results, hand_index=i)
        #         if hand_lm_data:
        #             print(f"Hand {i} Wrist (Landmark 0): x={hand_lm_data.landmark[0].x:.2f}, y={hand_lm_data.landmark[0].y:.2f}")


        cv.imshow('HandTracker Module Test', annotated_image)
        if cv.waitKey(5) & 0xFF == ord('q'):
            break

    tracker.close()
    cap.release()
    cv.destroyAllWindows()