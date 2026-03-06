#!/usr/bin/env python3
"""
Simple script to train on the kick_ball environment and render results.

Usage:
    python train_kick_ball.py
"""

import os
import sys

# Set environment path
ENV_PATH = "/home/jovyan/omni-epic/omni-epic/scenarios/generated/kick_ball/env_0.py"
OUTPUT_DIR = "omni-epic/scenarios/runs/kick_ball"
TRAIN_STEPS = 8000

print("="*80)
print("KICK BALL TRAINING")
print("="*80)
print(f"\nEnvironment: {ENV_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Training steps: {TRAIN_STEPS}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verify environment loads
print("\n" + "="*80)
print("STEP 1: VERIFYING ENVIRONMENT")
print("="*80)

print(f"\n[VERIFY] Testing environment...")
try:
    from embodied.envs.pybullet import PyBullet
    test_env = PyBullet(env_path=ENV_PATH, vision=False)
    test_env._env.reset()
    print(f"[VERIFY] ✓ Environment loads successfully!")
    test_env._env.close()
except Exception as e:
    print(f"[VERIFY] ✗ Environment failed to load: {e}")
    sys.exit(1)

# Train using command line to avoid config issues
print("\n" + "="*80)
print("STEP 2: TRAINING AGENT")
print("="*80)

dreamer_dir = os.path.join(OUTPUT_DIR, 'dreamer')

print(f"\n[TRAIN] Starting Dreamer training...")
print(f"[TRAIN] This will take a while ({TRAIN_STEPS} steps)...")
print(f"[TRAIN] Log directory: {dreamer_dir}")

# Change to omni-epic directory and run training
original_dir = os.getcwd()
os.chdir('/home/jovyan/omni-epic')

# Use command line arguments to override config
# Using smaller batch sizes and memory-efficient settings for vision
cmd = f"""python main_dreamer.py \
    hydra.job.chdir=False \
    logdir={dreamer_dir} \
    env.path={ENV_PATH} \
    env.vision=True \
    env.size=[32,32] \
    run.steps={TRAIN_STEPS} \
    run.eval_every=10000 \
    run.eval_eps=4 \
    run.train_ratio=256 \
    run.num_envs=4 \
    run.num_envs_eval=4 \
    run.driver_parallel=False \
    run.agent_process=False \
    run.remote_replay=False \
    run.log_every=30 \
    batch_size=32 \
    batch_length=64 \
    batch_length_eval=64 \
    jax.platform=gpu \
    jax.prealloc=False"""

print(f"\n[TRAIN] Running command from /home/jovyan/omni-epic:")
print(f"  {cmd}")

exit_code = os.system(cmd)

# Change back to original directory
os.chdir(original_dir)

if exit_code != 0:
    print(f"\n[TRAIN] ✗ Training failed with exit code {exit_code}")
    sys.exit(1)

print(f"\n[TRAIN] ✓ Training complete!")

# Render results
print("\n" + "="*80)
print("STEP 3: RENDERING TRAINED AGENT")
print("="*80)

import glob
from PIL import Image
import cv2

eval_dir = os.path.join(dreamer_dir, 'eval')
video_files = glob.glob(os.path.join(eval_dir, 'render*.mp4'))

if video_files:
    print(f"\n[RENDER] Found {len(video_files)} evaluation videos")
    print(f"[RENDER] Videos are located in: {eval_dir}")
    
    # Convert first video to GIF
    video_file = video_files[0]
    print(f"\n[RENDER] Converting video to GIF: {os.path.basename(video_file)}")
    
    # Extract frames from video
    cap = cv2.VideoCapture(video_file)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
        duration=50,
        loop=0
    )
    
    print(f"[RENDER] ✓ GIF saved to: {gif_path}")
else:
    print(f"[RENDER] ✗ No evaluation videos found in {eval_dir}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"  - Training logs: {dreamer_dir}")
if video_files:
    print(f"  - Trained agent GIF: {gif_path}")
