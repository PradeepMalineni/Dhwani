# File: train_model.py

import numpy as np
import json # For loading labels map
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
LABELS_JSON_FILE = "labels.json" # Path to your JSON label map
INPUT_DATA_X_FILE = "X_multisign_data.npy" # Should match output from preprocess_data.py
INPUT_DATA_Y_FILE = "y_multisign_data.npy" # Should match output from preprocess_data.py
MODEL_SAVE_PATH = "multisign_model.pkl" # New model name

def load_labels_map_inv(json_path):
    try:
        with open(json_path, 'r') as f:
            labels_map = json.load(f)
        # Create the inverse map (int_label: str_label)
        labels_map_inv = {v: k for k, v in labels_map.items()}
        print(f"Loaded inverse labels map: {labels_map_inv}")
        return labels_map_inv
    except Exception as e:
        print(f"Error loading or creating inverse labels map from {json_path}: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred, classes_names): # Renamed classes to classes_names
    cm = confusion_matrix(y_true, y_pred, labels=sorted(list(set(y_true) | set(y_pred)))) # Ensure all labels are considered
    plt.figure(figsize=(len(classes_names)*2, len(classes_names)*1.5)) # Adjust size based on num classes
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes_names, yticklabels=classes_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def main():
    labels_map_inv = load_labels_map_inv(LABELS_JSON_FILE)
    if not labels_map_inv:
        return

    try:
        X = np.load(INPUT_DATA_X_FILE)
        y = np.load(INPUT_DATA_Y_FILE)
        print(f"Successfully loaded X_data. Shape: {X.shape}")
        print(f"Successfully loaded y_data. Shape: {y.shape}")
        print(f"All unique numerical labels in loaded y_data: {np.unique(y)}")
    except FileNotFoundError:
        print(f"Error: Data files '{INPUT_DATA_X_FILE}' or '{INPUT_DATA_Y_FILE}' not found. Run preprocess_data.py.")
        return
    # ... (rest of your initial data checks from previous train_model.py)

    if X.shape[0] == 0: print("Error: Loaded X data is empty."); return

    print(f"--- Model Training Started ---")
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each.")
    class_distribution_str = {labels_map_inv.get(val, f"Unknown_{val}"): count
                              for val, count in zip(*np.unique(y, return_counts=True))}
    print(f"Class distribution: {class_distribution_str}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"\nTraining samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Unique labels in y_train: {np.unique(y_train)}")
    print(f"Unique labels in y_test: {np.unique(y_test)}")

    print("\nTraining Support Vector Machine (SVM)...") # Default model
    model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42, class_weight='balanced')
    # Or try RandomForest:
    # model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")

    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report (Test Set):")
    # Get all unique numerical labels present in either y_test or y_pred_test, sorted
    # These are the actual numerical labels the report will be based on.
    present_numerical_labels = sorted(list(set(y_test) | set(y_pred_test)))
    target_names_for_report = [labels_map_inv.get(label, f"Unknown_{label}") for label in present_numerical_labels]
    
    print(f"Debug: Numerical labels for report: {present_numerical_labels}")
    print(f"Debug: Target names for report: {target_names_for_report}")

    # Ensure 'labels' parameter in classification_report matches the unique labels found.
    print(classification_report(y_test, y_pred_test, labels=present_numerical_labels, target_names=target_names_for_report, zero_division=0))

    # For confusion matrix, use the same target_names_for_report
    plot_confusion_matrix(y_test, y_pred_test, classes_names=target_names_for_report)

    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\nTrained model saved to: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()