import json
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from tqdm import tqdm
from video_utils import load_clip

# ==================== Paths ====================
PARENT = Path(r'G:\Projects\Python\SignLanguage')
VIDEO_DIR    = PARENT / Path('Dataset/videos')
KEYPOINTS_DIR = PARENT / Path('Dataset/keypoints')
NSLT_FILE    = PARENT / Path('Dataset/nslt_300.json')
HAND_MODEL   = PARENT / Path('Dataset/landmarkers/hand_landmarker.task')
POSE_MODEL   = PARENT / Path('Dataset/landmarkers/pose_landmarker_lite.task')

KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Feature dims ====================
# Each frame's keypoints consist of:
# - 21 left hand landmarks, each with x, y, z coordinates (21 * 3 = 63)
# - 21 right hand landmarks, each with x, y, z coordinates (21 * 3 = 63)
# - 33 pose landmarks, each with x, y, z coordinates (33 * 3 = 99)
# Total keypoints per frame: 63 + 63 + 99 = 225
FEATURE_DIM = 225

# ==================== Build detectors ====================
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
)
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL)),
    running_mode=vision.RunningMode.IMAGE,
)

def extract_keypoints(frame_rgb: np.ndarray) -> np.ndarray:
    """Extract hand + pose keypoints from a single RGB frame.
    
    Returns: np.ndarray of shape (225,) — zeros where landmarks not detected.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # --- Hands ---
    hand_result = hand_detector.detect(mp_image)
    left_hand  = np.zeros(63)
    right_hand = np.zeros(63)

    for i, handedness in enumerate(hand_result.handedness):
        label = handedness[0].category_name  # 'Left' or 'Right'
        landmarks = hand_result.hand_landmarks[i]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()  # (63,)
        if label == 'Left':
            left_hand = coords
        else:
            right_hand = coords

    # --- Pose ---
    pose_result = pose_detector.detect(mp_image)
    pose = np.zeros(99)
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        pose = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()  # (99,)

    return np.concatenate([left_hand, right_hand, pose])  # (225,)


def extract_features(clip: np.ndarray) -> np.ndarray:
    """Extract and normalize pose and hand keypoints from all frames in a video clip.
    Args:
        clip: Video clip as a NumPy array of shape (n_frames, height, width, 3)
              containing uint8 RGB frames. Typically 32 frames of 224*224 pixels.
    
    Returns:
        Normalized keypoint features as a float32 array of shape (n_frames, 225).
        Each row corresponds to one frame, containing 225 landmark coordinates
        (63 left hand + 63 right hand + 99 pose) normalized to have mean 0 and
        standard deviation 1 within that frame.
    
    Note:
        Missing landmarks (e.g., hands not visible) are represented as zeros
        in the raw keypoints before normalization.
    """
    # Extract raw keypoints for every frame in the clip
    # Each call to extract_keypoints() returns a (225,) array
    features = np.array([extract_keypoints(frame) for frame in clip], dtype=np.float32)
    
    # Apply per-frame z-score normalization to reduce inter-frame variation
    mean = features.mean(axis=1, keepdims=True)  # (n_frames, 1)
    std  = features.std(axis=1, keepdims=True)   # (n_frames, 1)
    std  = np.where(std == 0, 1e-8, std)         # Prevent division by zero for static frames
    features = (features - mean) / std
    
    return features


if __name__ == '__main__':
    # Load the NSLT dataset metadata from JSON file
    with open(NSLT_FILE) as f:
        nslt = json.load(f)

    # Initialize counters for total videos, processed videos, and errors
    total, done, errors = len(nslt), 0, []

    # Create hand and pose detectors using MediaPipe options
    with vision.HandLandmarker.create_from_options(hand_options) as hand_detector, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_detector:

        # Process each video in the dataset with a progress bar
        for video_id, meta in tqdm(nslt.items(), desc="Processing videos"):
            # Define output path for features and input video path
            out_path   = KEYPOINTS_DIR / f'{video_id}.npy'
            video_path = VIDEO_DIR / f'{video_id}.mp4'

            # Skip if features already exist
            if out_path.exists():
                done += 1
                continue

            # Extract frame range from metadata
            frame_start = meta['action'][1]
            frame_end   = meta['action'][2]

            try:
                # Load the video clip for the specified frame range
                clip     = load_clip(video_path, frame_start, frame_end)  # (32, 224, 224, 3)
                # Extract keypoints for each frame in the clip
                features = extract_features(clip)                          # (32, 225)
                # Save the features as a NumPy array
                np.save(out_path, features)
            except Exception as e:
                # Record any errors that occur during processing
                errors.append((video_id, str(e)))
                print(f"error on {video_id}: {e}")

            done += 1
            os.system('cls' if os.name == 'nt' else 'clear')

    # Print summary of processing results
    print(f"\ndone. errors: {len(errors)}")
    if errors:
        for vid, err in errors:
            print(f"  {vid}: {err}")