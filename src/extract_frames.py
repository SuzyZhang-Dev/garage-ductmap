import os
import cv2
from pathlib import Path

def extract_frames(video_path, output_dir, fps_target=2):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

  step = max(int(round(fps / fps_target)), 1)

  i =0
  saved = 0
  while True:
    ok, frame = cap.read()
    if not ok:
      break
    if i % step == 0:
      output_path = os.path.join(output_dir, f'frame_{saved:05d}.jpg')
      cv2.imwrite(output_path, frame)
      saved += 1
    i += 1

  cap.release()
  print(f"Extracted {saved} frames to {output_dir}")
  return saved
  
if __name__ == "__main__":
  VIDEO_PATH = 'data/test03.mp4'
  OUTPUT_DIR = 'output/frames'
  FPS_TARGET = 2

  extract_frames(VIDEO_PATH, OUTPUT_DIR, FPS_TARGET)