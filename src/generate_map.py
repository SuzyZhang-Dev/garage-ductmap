import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def generate_map(trajectory, detection_results, output_path='output/maps/final_map.png'):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating map...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 1. Draw camera trajectory
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 
           'b-', linewidth=3, alpha=0.7, label='Camera Path')
    
    # Mark start and end points
    ax.plot(trajectory[0, 0], trajectory[0, 1], 
           'go', markersize=20, label='Start', zorder=10)
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 
           'ro', markersize=20, label='End', zorder=10)
    
    # 2. Draw detected wall lines
    # Draw every N frames (to avoid overcrowding)
    draw_interval = max(len(detection_results) // 20, 1)
    
    for i, frame_data in enumerate(detection_results):
        if i % draw_interval != 0:
            continue
        
        if frame_data['num_lines'] == 0:
            continue
        
        # Camera position
        cam_x, cam_y = trajectory[i]
        
        # Draw detected lines (simplified version: extending from camera position)
        for line in frame_data['lines'][:5]:  # Only take top 5 main lines
            angle = np.radians(line['angle'])
            length = min(line['length'] / 5, 100)  # Scale and limit length
            
            # Extend from camera position
            end_x = cam_x + length * np.cos(angle)
            end_y = cam_y + length * np.sin(angle)
            
            ax.plot([cam_x, end_x], [cam_y, end_y], 
                   'gray', alpha=0.2, linewidth=1)
        
        # Mark camera position
        ax.plot(cam_x, cam_y, 'b.', markersize=5, alpha=0.5)
    
    # 3. Beautify chart
    ax.set_xlabel('X Displacement (pixels)', fontsize=12)
    ax.set_ylabel('Y Displacement (pixels)', fontsize=12)
    ax.set_title('Ventilation Duct Map\\nCamera Trajectory + Detected Wall Lines', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')
    
    # Add statistics info
    total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    info_text = f"Frames: {len(detection_results)} | Total Distance: {total_distance:.0f}px"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Map saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    # Load data
    trajectory = np.load('output/trajectory/trajectory.npy')
    with open('output/detected/results.json', 'r') as f:
        results = json.load(f)
    
    generate_map(trajectory, results, 'output/maps/final_map.png')