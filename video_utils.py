import cv2
import numpy as np
from pathlib import Path

def load_clip(video_path: str | Path, frame_start: int, frame_end: int, n_frames=32) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # resolve -1 to last frame
    if frame_end == -1:
        frame_end = total_frames

    # convert from 1-indexed to 0-indexed and clamp
    frame_start = max(0, frame_start - 1)
    frame_end   = min(frame_end, total_frames)

    # uniformly sample n_frames indices within the clip
    indices = np.linspace(frame_start, frame_end - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()
    return np.array(frames)  # (n_frames, 224, 224, 3)


if __name__ == '__main__':
    import json

    JSON_FILE = Path(r'G:\Projects\Python\SignLanguage\Dataset\nslt_300.json')
    with open(JSON_FILE) as f:
        nslt = json.load(f)

    for idx, (video_id, meta) in enumerate(nslt.items(), 1):
        frame_start = meta['action'][1]
        frame_end   = meta['action'][2]
        video_path  = f'videos/{video_id}.mp4'

        print(f"\n[{idx:03d}] video:       {video_path}")
        print(f"     frame range: {frame_start} → {frame_end}")

        clip = load_clip(video_path, frame_start, frame_end)

        print(f"     clip shape:  {clip.shape}")       # (32, 224, 224, 3)
        print(f"     dtype:       {clip.dtype}")        # uint8
        print(f"     min/max:     {clip.min()} / {clip.max()}")
