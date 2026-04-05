import json
import cv2
import numpy as np
from pathlib import Path

NSLT_PATH = Path(r'G:\Projects\Python\SignLanguage\Dataset\nslt_300.json')
with open(NSLT_PATH) as f:
    nslt = json.load(f)

durations = []
for video_id, meta in nslt.items():
    frame_start = meta['action'][1]
    frame_end   = meta['action'][2]
    
    if frame_end == -1:
        cap = cv2.VideoCapture(f'videos/{video_id}.mp4')
        frame_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    duration_frames = frame_end - frame_start
    duration_secs   = duration_frames / 25
    durations.append(duration_frames)

durations = np.array(durations)
print(f"avg frames:  {durations.mean():.1f}")
print(f"median:      {np.median(durations):.1f}")
print(f"min:         {durations.min()}")
print(f"max:         {durations.max()}")
print(f"avg seconds: {durations.mean()/25:.2f}s")