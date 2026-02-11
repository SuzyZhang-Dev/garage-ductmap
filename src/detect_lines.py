import cv2
import numpy as np
import math
import json
from pathlib import Path

class PipelineConfig:
  def __init__(self,image_path,resize_width=960):
    self.image_path = image_path
    self.resize_width = resize_width

    self.img = cv2.imread(self.image_path)
    if self.img is None:
      raise RuntimeError(f"Could not read image: {self.image_path}")
    
    self.img = self._resize_image(self.img)
    gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    self.gray = gray

    self.brightness = gray.mean()
    self.contrast = gray.std()
    self.median = np.median(gray)

    self._set_adaptive_params()

  def _resize_image(self, img):
    height,width = img.shape[:2]
    if width > self.resize_width:
      scale = self.resize_width / width
      new_size = (int(width * scale), int(height * scale))
      return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img
  
  def _set_adaptive_params(self):
    # 20-40 low contrast(cloudy or night)
    # 40-70 normal contrast
    # 70-100 high contrast(sunny or shady)
    if self.contrast < 30:
      canny_scale = 0.7
      hough_scale = 1.3
    elif self.contrast > 80:
      canny_scale = 1.2
      hough_scale = 0.9
    else:
      canny_scale = 1.0
      hough_scale = 1.0

    sigma = 0.33
    self.canny1 = int(max(10, (1 - sigma) * self.median *canny_scale))
    self.canny2 = int(min(255, (1 + sigma) * self.median * canny_scale))
    
    base_hough = int(self.img.shape[1] * 0.1)
    self.hough_threshold = int(base_hough * hough_scale)

    self.min_line_length = 100
    self.max_line_gap = 10

def detect_lines(config):
  edges = cv2.Canny(config.gray, config.canny1, config.canny2, L2gradient=True)
  lines = cv2.HoughLinesP(edges, 1, np.pi / 180, config.hough_threshold, minLineLength=config.min_line_length, maxLineGap=config.max_line_gap)
  lines = filter_lines_by_main_angle(lines, threshold=12, bin_deg=2, min_ratio=0.25)
  return edges, lines


def filter_lines_by_main_angle(lines, threshold=20, bin_deg=2, min_ratio=0.25):
  if lines is None or len(lines) == 0:
    return lines
  angles = []
  lengths = []
  for x1, y1, x2, y2 in lines[:, 0]:
    dx, dy = x2 - x1, y2 - y1
    angle = (math.degrees(math.atan2(dy, dx)) + 180) % 180
    length = math.hypot(dx,dy)
    angles.append(angle)
    lengths.append(length)
  angles = np.array(angles)
  lengths = np.array(lengths)

  bins = np.arange(0, 180 + bin_deg, bin_deg)
  hist, edges = np.histogram(angles, bins=bins, weights=lengths)
  order = np.argsort(hist)[::-1]
  if hist[order[0]]<= 1e-6:
    return None
  
  peaks = []
  strongest = hist[order[0]]  
  for i in order:
    if hist[i] < strongest * min_ratio:
      break
    center = (edges[i] + edges[i + 1]) / 2
    if all(min(abs(center-p), 180 - abs(center - p)) > threshold *2 for p in peaks):
      peaks.append(center)
    if len(peaks) >=2:
      break
           

  def angle_diff(a, b):
    diff = abs(a - b)
    return min(diff, 180 - diff)
  
  keep = [i for i, a in enumerate(angles) if any(angle_diff(a, p) <= threshold for p in peaks)]
  if not keep:
    return None
  return lines[keep]


def process_all_frames(frame_dir, output_dir):
  frame_dir = Path(frame_dir)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  frame_paths = sorted(frame_dir.glob("*.jpg"))
  if not frame_paths:
    raise RuntimeError(f"No frames found in {frame_dir}")
  
  results = []
  for i, frame_path in enumerate(frame_paths):
    config = PipelineConfig(frame_path)
    edges, lines = detect_lines(config)
    frame_result = {
      'frame_id': i,
      'frame_name': frame_path.name,
      'img_size':config.img.shape[:2],
      'brightness': float(config.brightness),
      'contrast': float(config.contrast),
      'num_lines': int(len(lines)) if lines is not None else 0,
      'lines':[]
    }

    if lines is not None:
      for x1,y1,x2,y2 in lines[:,0]:
        angle = math.degrees(math.atan2(y2-y1, x2-x1))
        length = math.hypot(x2-x1, y2-y1)
        frame_result['lines'].append({'x1':int(x1), 
                                      'y1':int(y1), 
                                      'x2':int(x2), 
                                      'y2':int(y2), 
                                      'angle':float(angle), 
                                      'length':float(length)})
    results.append(frame_result)

    vis = config.img.copy()
    if lines is not None:
      for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    vis_path = output_dir / f"detected_{frame_path.name}"
    cv2.imwrite(str(vis_path), vis)

  json_path = output_dir / "results.json"
  with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

  line_counts = [r['num_lines'] for r in results]
  print(f"Processed {len(results)} frames. Line counts: min={min(line_counts)}, max={max(line_counts)}, avg={sum(line_counts)/len(line_counts):.2f}")

  return results


if __name__ == "__main__":
  results = process_all_frames('output/frames', 'output/detected')