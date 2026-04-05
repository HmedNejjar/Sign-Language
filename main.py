#type: ignore
import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from pathlib import Path

from model import SignGRU
from plot_metrics import MetricsLogger   # for post-training curve display

# ==================== Config ====================
PARENT = Path(r'G:\Projects\Python\SignLanguage\Dataset')
CLASS_PATH = PARENT / Path('wlasl_class_list.txt')
MODEL_PATH   = Path('models/SignLang_model.pth')
N_FRAMES     = 32       # frames to buffer before predicting
INPUT_SIZE   = 225      # 21*3 (hand) + 33*3 (pose) + 21*3 (other hand) = 225
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
NUM_CLASSES  = 300
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== MediaPipe Setup ====================
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

def extract_keypoints(results) -> np.ndarray:
    """
    Extracts landmarks and flattens them into a single 1D array.
    Matches the 225 input size: pose(99) + left_hand(63) + right_hand(63).
    """
    pose = (
        np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 3)
    )
    lh = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, lh, rh])   # (225,)


# ==================== Load Class List ====================
def load_classes(path: str | Path, num_classes: int) -> list[str]:
    """Load sign class names from the tab-separated class list file.
    Args:
        path:        Path to wlasl_class_list.txt
        num_classes: How many classes to load (matches NUM_CLASSES used at training time).

    Returns:
        List of class-name strings, indexed 0 … num_classes-1.

    Note:
        We sort by the integer index in column 0 rather than relying on file order,
        which makes the mapping robust to re-ordered files.
    """
    entries: list[tuple[int, str]] = []
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

    # Sort by index so position 0 → class 0, position 1 → class 1, etc.
    entries.sort(key=lambda x: x[0])

    # Keep only the first num_classes entries (indices 0 … num_classes-1)
    classes = [word for _, word in entries[:num_classes]]

    if len(classes) < num_classes:
        raise ValueError(
            f"Requested {num_classes} classes but only {len(classes)} found in {path}"
        )
    return classes


CLASSES = load_classes(CLASS_PATH, NUM_CLASSES)


# ==================== Load Model ====================
model = SignGRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# ==================== Optional: Plot Training Curves ====================
def show_training_curves(metrics_path: str | Path = 'metrics\\metrics.json') -> None:
    """Display saved training curves (loss + accuracy) if metrics.json exists.

    Call this before or after the webcam loop, e.g.:
        show_training_curves()
    """
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        print(f"[plot] No metrics file found at {metrics_path} — skipping plot.")
        return
    logger = MetricsLogger.load(metrics_path)
    logger.plot(save_path='training_curves.png', show=True)


# ==================== Inference Loop ====================
cap          = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=N_FRAMES)
current_pred = "..."
current_conf = 0.0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (360, 640))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        keypoints = extract_keypoints(results)
        frame_buffer.append(keypoints)

        if len(frame_buffer) == N_FRAMES:
            session_input = np.expand_dims(list(frame_buffer), axis=0)   # (1, 32, 225)
            x = torch.tensor(session_input, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

                current_pred = CLASSES[pred_idx.item()]
                current_conf = conf.item()

        # HUD overlay
        cv2.rectangle(image, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(image, f'{current_pred} ({current_conf:.2%})', (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks,       mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,  mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('Sign Language GRU', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()