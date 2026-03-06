#!/usr/bin/env python3
"""
Quick script to:
1. Generate a single environment
2. Train an agent on it
3. Render the trained agent as a GIF

Usage:
    python quick_train_and_render.py
"""

import os
import json
from omegaconf import OmegaConf
from omni_epic.robots import robot_dict
from omni_epic.core.fm import FM
from main_dreamer import main_dreamer
from PIL import Image
import numpy as np

# Configuration
ROBOT = "r2d2"
TASK_DESC = robot_dict[ROBOT]["task_descs_init"][0]  # Use first seeded task
OUTPUT_DIR = "omni-epic/quick_test"
TRAIN_STEPS = 100000  # Number of training steps (adjust as needed)

print("="*80)
print("QUICK TRAIN AND RENDER")
print("="*80)
print(f"\nRobot: {ROBOT}")
print(f"Task: {TASK_DESC[:150]}...")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Training steps: {TRAIN_STEPS}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Generate environment code
print("\n" + "="*80)
print("STEP 1: GENERATING ENVIRONMENT CODE")
print("="*80)

# Load config for environment generator
config_env_gen = OmegaConf.create({
    'client': 'openai',
    'model': 'claude-4-5-sonnet-latest',
    'max_tokens': 4096,
    'temperature': 0
})

fm_env_generator = FM(config_env_gen)

print(f"\n[GEN] Querying LLM for environment code...")
env_code = fm_env_generator.query_env_code(ROBOT, TASK_DESC, add_examples=True)

# Iterate on compilation errors (max 5 iterations)
MAX_ERROR_ITERATIONS = 5
gen_num = fm_env_generator.iterate_on_errors(
    ROBOT,
    TASK_DESC,
    env_code,
    OUTPUT_DIR,
    add_examples=True,
    env_paths_other=[],
    iteration_max=MAX_ERROR_ITERATIONS,
)

if gen_num < 0:
    print(f"\n[GEN] ✗ Failed to generate valid environment code after {MAX_ERROR_ITERATIONS} iterations")
    print(f"[GEN] Exiting.")
    exit(1)

# Use the successfully generated environment
env_path = os.path.join(OUTPUT_DIR, f'env_{gen_num}.py')
print(f"\n[GEN] ✓ Environment code generated successfully after {gen_num} iterations")
print(f"[GEN] Environment saved to: {env_path}")

# Test if environment loads
print(f"\n[GEN] Testing environment...")
try:
    from embodied.envs.pybullet import PyBullet
    test_env = PyBullet(env_path=env_path, vision=Tr)
    test_env._env.reset()
    print(f"[GEN] ✓ Environment loads successfully!")
    test_env._env.close()
except Exception as e:
    print(f"[GEN] ✗ Environment failed to load: {e}")
    print(f"[GEN] This shouldn't happen after error iteration. Exiting.")
    exit(1)

# Step 2: Train agent
print("\n" + "="*80)
print("STEP 2: TRAINING AGENT")
print("="*80)

dreamer_dir = os.path.join(OUTPUT_DIR, 'dreamer')

# Create Dreamer config
config_dreamer = OmegaConf.create({
    'logdir': dreamer_dir,
    'env': {
        'path': env_path,
    },
    'run': {
        'steps': TRAIN_STEPS,
        'eval_every': 10000,
        'eval_eps': 1,
        'train_ratio': 512,
        'from_checkpoint': '',
    },
    'jax': {
        'platform': 'cpu',  # Change to 'gpu' if you have GPU
        'precision': 'float32',
    },
    'batch_size': 16,
    'batch_length': 64,
})

print(f"\n[TRAIN] Starting Dreamer training...")
print(f"[TRAIN] This will take a while ({TRAIN_STEPS} steps)...")
main_dreamer(config_dreamer)
print(f"[TRAIN] ✓ Training complete!")

# Step 3: Render trained agent
print("\n" + "="*80)
print("STEP 3: RENDERING TRAINED AGENT")
print("="*80)

# The trained agent videos should be in dreamer_dir/eval/
eval_dir = os.path.join(dreamer_dir, 'eval')

# Check if eval videos exist
import glob
video_files = glob.glob(os.path.join(eval_dir, 'render*.mp4'))

if video_files:
    print(f"\n[RENDER] Found {len(video_files)} evaluation videos")
    print(f"[RENDER] Videos are located in: {eval_dir}")
    
    # Convert first video to GIF
    video_file = video_files[0]
    print(f"\n[RENDER] Converting video to GIF: {os.path.basename(video_file)}")
    
    # Extract frames from video
    import cv2
    cap = cv2.VideoCapture(video_file)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    
    print(f"[RENDER] Extracted {len(frames)} frames")
    
    # Save as GIF
    gif_path = os.path.join(OUTPUT_DIR, 'trained_agent.gif')
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=50,  # 50ms per frame = 20 FPS
        loop=0
    )
    
    print(f"[RENDER] ✓ GIF saved to: {gif_path}")
else:
    print(f"[RENDER] ✗ No evaluation videos found in {eval_dir}")
    print(f"[RENDER] Training may not have completed evaluation episodes")

# Save metadata
metadata = {
    'robot': ROBOT,
    'task_description': TASK_DESC,
    'env_path': env_path,
    'dreamer_dir': dreamer_dir,
    'train_steps': TRAIN_STEPS,
}

with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"  - Environment code: {env_path}")
print(f"  - Training logs: {dreamer_dir}")
print(f"  - Trained agent GIF: {os.path.join(OUTPUT_DIR, 'trained_agent.gif')}")
print("\nYou can now view the GIF to see how the trained agent performs!")
