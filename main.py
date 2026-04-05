# type: ignore
import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from pathlib import Path
from model import SignGRU

from train import HIDDEN_SIZE

# ==================== Config ====================
PARENT      = Path(r'G:\Projects\Python\SignLanguage\Dataset')
CLASS_PATH  = PARENT / 'wlasl_class_list.txt'
MODEL_PATH  = Path('models/SignLang_model.pth')
HAND_MODEL  = Path('hand_landmarker.task')
POSE_MODEL  = Path('pose_landmarker_lite.task')

N_FRAMES    = 32
INPUT_SIZE  = 225
NUM_LAYERS  = 2
NUM_CLASSES = 300
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== Load Class List ====================
def load_classes(path: str | Path, num_classes: int) -> list[str]:
    entries = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            idx, word = int(parts[0]), parts[1].strip()
            entries.append((idx, word))
    entries.sort(key=lambda x: x[0])
    classes = [word for _, word in entries[:num_classes]]
    if len(classes) < num_classes:
        raise ValueError(f"Requested {num_classes} classes but only {len(classes)} found")
    return classes

CLASSES = load_classes(CLASS_PATH, NUM_CLASSES)


# ==================== Load Model ====================
model = SignGRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    print("model loaded.")
else:
    print("WARNING: no model found, running with random weights.")
model.eval()


# ==================== MediaPipe Setup ====================
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
)
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL)),
    running_mode=vision.RunningMode.IMAGE,
)


def extract_keypoints(frame_rgb: np.ndarray,
                      hand_detector,
                      pose_detector) -> np.ndarray:
    """Extract keypoints matching precompute.py order: [left_hand, right_hand, pose]"""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # --- Hands ---
    hand_result = hand_detector.detect(mp_image)
    left_hand   = np.zeros(63)
    right_hand  = np.zeros(63)

    for i, handedness in enumerate(hand_result.handedness):
        label     = handedness[0].category_name  # 'Left' or 'Right'
        landmarks = hand_result.hand_landmarks[i]
        coords    = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        if label == 'Left':
            left_hand = coords
        else:
            right_hand = coords

    # --- Pose ---
    pose_result = pose_detector.detect(mp_image)
    pose = np.zeros(99)
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        pose = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

    # order must match precompute.py: [left_hand, right_hand, pose]
    return np.concatenate([left_hand, right_hand, pose])  # (225,)


def normalize_frame(frame_features: np.ndarray) -> np.ndarray:
    """Per-frame z-score normalization — must match precompute.py."""
    mean = frame_features.mean()
    std  = frame_features.std()
    std  = std if std > 1e-8 else 1e-8
    return (frame_features - mean) / std


# ==================== Inference Loop ====================
cap          = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=N_FRAMES)
current_pred = "..."
current_conf = 0.0

with vision.HandLandmarker.create_from_options(hand_options) as hand_detector, \
     vision.PoseLandmarker.create_from_options(pose_options) as pose_detector:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (640, 480))

        # extract + normalize — must match precompute pipeline
        keypoints = extract_keypoints(frame_rgb, hand_detector, pose_detector)
        keypoints = normalize_frame(keypoints)
        frame_buffer.append(keypoints)

        if len(frame_buffer) == N_FRAMES:
            x = torch.tensor(
                np.expand_dims(list(frame_buffer), axis=0),
                dtype=torch.float32
            ).to(DEVICE)  # (1, 32, 225)

            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                current_pred = CLASSES[pred_idx.item()]
                current_conf = conf.item()

        # HUD overlay
        display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(display, (0, 0), (350, 40), (245, 117, 16), -1)
        cv2.putText(display, f'{current_pred} ({current_conf:.2%})', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()