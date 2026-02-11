#!/usr/bin/env python3
"""
Ventilation Duct Mapping System
Generate 2D maps from video
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from extract_frames import extract_frames
from detect_lines import process_all_frames
from track_motion import estimate_camera_motion
from generate_map import generate_map


def main():
    print("=" * 70)
    print("Ventilation Duct Mapping System".center(70))
    print("=" * 70)
    
    # Configuration
    VIDEO_PATH = 'data/test03.mp4'
    FRAMES_DIR = 'output/frames'
    DETECTED_DIR = 'output/detected'
    TRAJECTORY_DIR = 'output/trajectory'
    MAP_PATH = 'output/maps/final_map.png'
    
    # Step 1: Extract frames
    print("\n[Step 1/4] Extracting video frames")
    print("-" * 70)
    num_frames = extract_frames(VIDEO_PATH, FRAMES_DIR, fps_target=2)
    
    # Step 2: Detect lines
    print("\n[Step 2/4] Detecting duct lines")
    print("-" * 70)
    results = process_all_frames(FRAMES_DIR, DETECTED_DIR)
    
    # Step 3: Track camera motion
    print("\n[Step 3/4] Tracking camera trajectory")
    print("-" * 70)
    trajectory = estimate_camera_motion(FRAMES_DIR, TRAJECTORY_DIR)
    
    # Step 4: Generate map
    print("\n[Step 4/4] Generating 2D map")
    print("-" * 70)
    generate_map(trajectory, results, MAP_PATH)
    
    # Summary
    print("\n" + "=" * 70)
    print("Processing complete!".center(70))
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  ├─ Frames: {FRAMES_DIR}/ ({num_frames} images)")
    print(f"  ├─ Detection results: {DETECTED_DIR}/")
    print(f"  ├─ Trajectory data: {TRAJECTORY_DIR}/trajectory.npy")
    print(f"  └─ Final map: {MAP_PATH}")
    print()


if __name__ == "__main__":
    main()