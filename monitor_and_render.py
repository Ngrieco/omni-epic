#!/usr/bin/env python3
"""
Monitor training progress and create GIFs from evaluation episodes in real-time.

Usage:
    # In one terminal, run training:
    python simple_train_and_render.py
    
    # In another terminal, run this monitor:
    python monitor_and_render.py
"""

import os
import time
import pickle
import glob
from pathlib import Path
from PIL import Image
import numpy as np

# Configuration
EVAL_DIR = "/home/jovyan/omni-epic/omni-epic/scenarios/runs/maze_checkpoints/dreamer/eval"
OUTPUT_DIR = "omni-epic/simple_test/gifs"
CHECK_INTERVAL = 10  # Check for new episodes every 10 seconds

print("="*80)
print("TRAINING MONITOR - GIF RENDERER")
print("="*80)
print(f"\nCurrent working directory: {os.getcwd()}")
print(f"Watching: {EVAL_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Check interval: {CHECK_INTERVAL} seconds")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Track processed files
processed_files = set()

def extract_step_from_filename(filename):
    """Extract step number from episode filename if possible."""
    # Filenames are like: episode_2024-01-01_123456_789012_250.pickle
    # The last number before .pickle is the episode length
    # We'll use timestamp to order them
    return os.path.getmtime(filename)

def create_gif_from_pickle(pickle_path, output_path_base):
    """Create GIFs from a pickle file containing episode data (both first-person and third-person)."""
    try:
        with open(pickle_path, 'rb') as f:
            episode_data = pickle.load(f)
        
        created_gifs = []
        
        # Check for first-person view (render)
        if 'policy_render' in episode_data:
            frames = episode_data['policy_render']
            if frames is not None and len(frames) > 0:
                images = []
                for frame in frames:
                    if isinstance(frame, np.ndarray):
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                        if len(frame.shape) == 2:
                            frame = np.stack([frame] * 3, axis=-1)
                        images.append(Image.fromarray(frame))
                
                if images:
                    output_path = output_path_base.replace('.gif', '_firstperson.gif')
                    images[0].save(
                        output_path,
                        save_all=True,
                        append_images=images[1:],
                        duration=50,
                        loop=0
                    )
                    created_gifs.append('first-person')
        
        # Check for third-person view (render3p)
        if 'policy_render3p' in episode_data:
            frames = episode_data['policy_render3p']
            if frames is not None and len(frames) > 0:
                images = []
                for frame in frames:
                    if isinstance(frame, np.ndarray):
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                        if len(frame.shape) == 2:
                            frame = np.stack([frame] * 3, axis=-1)
                        images.append(Image.fromarray(frame))
                
                if images:
                    output_path = output_path_base.replace('.gif', '_thirdperson.gif')
                    images[0].save(
                        output_path,
                        save_all=True,
                        append_images=images[1:],
                        duration=50,
                        loop=0
                    )
                    created_gifs.append('third-person')
        
        # Fallback to policy_image if neither render view is available
        if not created_gifs and 'policy_image' in episode_data:
            frames = episode_data['policy_image']
            if frames is not None and len(frames) > 0:
                images = []
                for frame in frames:
                    if isinstance(frame, np.ndarray):
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                        if len(frame.shape) == 2:
                            frame = np.stack([frame] * 3, axis=-1)
                        images.append(Image.fromarray(frame))
                
                if images:
                    images[0].save(
                        output_path_base,
                        save_all=True,
                        append_images=images[1:],
                        duration=50,
                        loop=0
                    )
                    created_gifs.append('image')
        
        if not created_gifs:
            print(f"  ⚠ No render data found in {os.path.basename(pickle_path)}")
            return False
        
        return created_gifs
        
    except Exception as e:
        print(f"  ✗ Error processing {os.path.basename(pickle_path)}: {e}")
        return False

def monitor_and_render():
    """Monitor eval directory and create GIFs from new episodes."""
    print("\n🔍 Monitoring started... (Press Ctrl+C to stop)\n")
    
    episode_count = 0
    
    try:
        while True:
            # Find all pickle files
            pickle_files = glob.glob(os.path.join(EVAL_DIR, "episode_*.pickle"))
            
            # Process new files
            new_files = [f for f in pickle_files if f not in processed_files]
            
            if new_files:
                # Sort by modification time
                new_files.sort(key=extract_step_from_filename)
                
                for pickle_file in new_files:
                    episode_count += 1
                    basename = os.path.basename(pickle_file)
                    
                    # Create output filename
                    gif_filename = f"episode_{episode_count:03d}.gif"
                    gif_path = os.path.join(OUTPUT_DIR, gif_filename)
                    
                    print(f"[Episode {episode_count}] Processing {basename}...")
                    
                    if create_gif_from_pickle(pickle_file, gif_path):
                        print(f"  ✓ GIF saved: {gif_filename}")
                    
                    processed_files.add(pickle_file)
            
            # Wait before next check
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("MONITORING STOPPED")
        print("="*80)
        print(f"\nProcessed {episode_count} episodes")
        print(f"GIFs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    # Wait for eval directory to be created
    print("\nWaiting for training to start...")
    while not os.path.exists(EVAL_DIR):
        time.sleep(2)
    
    print(f"✓ Found eval directory: {EVAL_DIR}\n")
    
    monitor_and_render()
