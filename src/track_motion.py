import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def estimate_camera_motion(frames_dir, output_dir=None):
    """
    Estimate camera motion trajectory using optical flow
    
    Returns:
        trajectory: Nx2 array, each row is (x, y) coordinate
    """
    frames_dir = Path(frames_dir)
    frame_paths = sorted(frames_dir.glob('frame_*.jpg'))
    
    if len(frame_paths) < 2:
        raise RuntimeError("Need at least 2 frames for motion tracking")
    
    print(f"\nStarting camera motion tracking ({len(frame_paths)} frames)...")
    
    # Read first frame
    prev_frame = cv2.imread(str(frame_paths[0]))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Trajectory (cumulative displacement)
    trajectory = [(0.0, 0.0)]
    
    # Feature detection parameters
    feature_params = dict(
        maxCorners=200,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    # Optical flow parameters
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    for i in range(1, len(frame_paths)):
        curr_frame = cv2.imread(str(frame_paths[i]))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect feature points
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        
        if p0 is not None and len(p0) > 10:
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
            
            # Filter good points
            if p1 is not None and st is not None:
                good_old = p0[st == 1]
                good_new = p1[st == 1]
                
                if len(good_old) > 10:
                    # Calculate median displacement (more robust than mean)
                    dx = np.median(good_new[:, 0] - good_old[:, 0])
                    dy = np.median(good_new[:, 1] - good_old[:, 1])
                    
                    # Accumulate trajectory
                    last_x, last_y = trajectory[-1]
                    trajectory.append((last_x + dx, last_y + dy))
                else:
                    trajectory.append(trajectory[-1])
            else:
                trajectory.append(trajectory[-1])
        else:
            trajectory.append(trajectory[-1])
        
        prev_gray = curr_gray
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(frame_paths)} ({(i+1)/len(frame_paths)*100:.1f}%)")
    
    trajectory = np.array(trajectory)
    
    print(f"\n✓ Trajectory tracking completed:")
    print(f"  Trajectory points: {len(trajectory)}")
    print(f"  Total displacement: {np.linalg.norm(trajectory[-1] - trajectory[0]):.1f} pixels")
    
    # Visualize trajectory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Camera Path')
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=15, label='Start')
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=15, label='End')
        
        # Mark every 10th frame
        for i in range(0, len(trajectory), 10):
            plt.plot(trajectory[i, 0], trajectory[i, 1], 'ko', markersize=3)
        
        plt.xlabel('X displacement (pixels)')
        plt.ylabel('Y displacement (pixels)')
        plt.title('Camera Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        traj_path = output_dir / 'trajectory.png'
        plt.savefig(traj_path, dpi=150, bbox_inches='tight')
        print(f"  Trajectory plot: {traj_path}")
        
        # Save data
        np.save(output_dir / 'trajectory.npy', trajectory)
    
    return trajectory


if __name__ == "__main__":
    trajectory = estimate_camera_motion('output/frames', 'output/trajectory')