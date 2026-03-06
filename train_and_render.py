#!/usr/bin/env python3
"""
Simplified script to:
1. Use an existing working environment OR generate from scenario
2. Train an agent on it
3. Render the trained agent as a GIF

Usage:
    python simple_train_and_render.py
    python simple_train_and_render.py --scenario scenarios/maze_checkpoints.md
"""

import os
import sys
import json
import re
from pathlib import Path
from omegaconf import OmegaConf
from main_dreamer import main_dreamer
from PIL import Image
import glob

# Parse command line arguments
def parse_scenario_md(md_path):
    """Parse scenario markdown file and extract task description."""
    with open(md_path, 'r') as f:
        content = f.read()
    
    # Extract task description
    task_match = re.search(r'## Task Description\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if task_match:
        return task_match.group(1).strip()
    return None

# Parse command line arguments
ENV_PATH = None
TASK_DESCRIPTION = None
SCENARIO_NAME = None

if len(sys.argv) > 1:
    if sys.argv[1] == '--env':
        # Use existing environment file directly
        if len(sys.argv) < 3:
            print("Error: --env requires a path to an environment file")
            print("Usage: python simple_train_and_render.py --env path/to/env.py")
            sys.exit(1)
        
        ENV_PATH = sys.argv[2]
        if not os.path.exists(ENV_PATH):
            print(f"Error: Environment file not found: {ENV_PATH}")
            sys.exit(1)
        
        # Extract scenario name from path if possible
        path_parts = Path(ENV_PATH).parts
        if 'generated' in path_parts:
            idx = path_parts.index('generated')
            if idx + 1 < len(path_parts):
                SCENARIO_NAME = path_parts[idx + 1]
        
        if not SCENARIO_NAME:
            SCENARIO_NAME = Path(ENV_PATH).stem
        
        print(f"Using existing environment: {ENV_PATH}")
    
    elif sys.argv[1] == '--scenario':
        # Generate environment from scenario
        if len(sys.argv) < 3:
            print("Error: --scenario requires a path to a markdown file")
            print("Usage: python simple_train_and_render.py --scenario scenarios/maze_checkpoints.md")
            sys.exit(1)
        
        scenario_path = sys.argv[2]
        if not os.path.exists(scenario_path):
            print(f"Error: Scenario file not found: {scenario_path}")
            sys.exit(1)
        
        TASK_DESCRIPTION = parse_scenario_md(scenario_path)
        SCENARIO_NAME = Path(scenario_path).stem
        
        if not TASK_DESCRIPTION:
            print(f"Error: Could not extract task description from {scenario_path}")
            sys.exit(1)
    else:
        print("Error: Unknown argument")
        print("Usage:")
        print("  python simple_train_and_render.py                                    # Use default environment")
        print("  python simple_train_and_render.py --env path/to/env.py              # Use existing environment")
        print("  python simple_train_and_render.py --scenario scenarios/task.md      # Generate from scenario")
        sys.exit(1)
else:
    # Default: use existing environment
    ENV_PATH = "omni_epic/envs/r2d2/cross_bridge.py"

# Set output directory based on scenario name
if SCENARIO_NAME:
    OUTPUT_DIR = f"omni-epic/scenarios/runs/{SCENARIO_NAME}"
else:
    OUTPUT_DIR = "omni-epic/simple_test"

TRAIN_STEPS = 100000  # Number of training steps (adjust as needed)

print("="*80)
print("SIMPLE TRAIN AND RENDER")
print("="*80)

# Step 0: Generate environment if using scenario
if TASK_DESCRIPTION:
    print(f"\nScenario: {SCENARIO_NAME}")
    print(f"Task: {TASK_DESCRIPTION[:100]}...")
    
    print("\n" + "="*80)
    print("STEP 0: GENERATING ENVIRONMENT FROM SCENARIO")
    print("="*80)
    
    from omni_epic.core.fm import FM
    from hydra import compose, initialize_config_dir
    
    # Load config
    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
        omni_config = compose(config_name="omni_epic")
    
    # Generate environment
    fm = FM(omni_config.environment_generator)
    
    # Create output directory for generated environment
    gen_output_dir = f"omni-epic/scenarios/generated/{SCENARIO_NAME}"
    os.makedirs(gen_output_dir, exist_ok=True)
    
    print(f"\n[GEN] Generating environment code from task description...")
    print(f"[GEN] This may take 30-120 seconds...")
    
    # Generate environment code
    completion = fm.query_env_code(
        'r2d2',
        TASK_DESCRIPTION,
        add_examples=omni_config.add_examples
    )
    
    print(f"[GEN] Checking for compilation errors...")
    
    gen_num = fm.iterate_on_errors(
        'r2d2',
        TASK_DESCRIPTION,
        completion,
        gen_output_dir,
        add_examples=omni_config.add_examples,
        env_paths_other=[],
        iteration_max=omni_config.error_max_iterations,
    )
    
    if gen_num < 0:
        print("[GEN] ✗ Failed to generate environment")
        sys.exit(1)
    
    ENV_PATH = os.path.abspath(os.path.join(gen_output_dir, f'env_{gen_num}.py'))
    OUTPUT_DIR = f"omni-epic/scenarios/runs/{SCENARIO_NAME}"
    
    print(f"[GEN] ✓ Environment generated: {ENV_PATH}")
    print(f"[GEN] Visualizations saved to: {gen_output_dir}")

print(f"\nEnvironment: {ENV_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Training steps: {TRAIN_STEPS}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Verify environment loads
print("\n" + "="*80)
print("STEP 1: VERIFYING ENVIRONMENT")
print("="*80)

print(f"\n[VERIFY] Testing environment...")
try:
    from embodied.envs.pybullet import PyBullet
    test_env = PyBullet(env_path=ENV_PATH, vision=True)
    test_env._env.reset()
    print(f"[VERIFY] ✓ Environment loads successfully!")
    test_env._env.close()
except Exception as e:
    print(f"[VERIFY] ✗ Environment failed to load: {e}")
    exit(1)

# Step 2: Train agent
print("\n" + "="*80)
print("STEP 2: TRAINING AGENT")
print("="*80)

dreamer_dir = os.path.join(OUTPUT_DIR, 'dreamer')

# Create Dreamer config - load from default config and override
from hydra import compose, initialize_config_dir
import os

# Get absolute path to config directory
config_dir = os.path.abspath('configs')

# Initialize Hydra with the config directory
with initialize_config_dir(config_dir=config_dir, version_base=None):
    # Load the default dreamer config
    config_dreamer = compose(config_name="dreamer/dreamer_xxs")
    
    # Allow struct modification
    OmegaConf.set_struct(config_dreamer, False)
    
    # Debug: print what keys are in the config
    print(f"\n[DEBUG] Config keys: {list(config_dreamer.keys())}")
    
    # The config is nested under 'dreamer' key - extract it
    if 'dreamer' in config_dreamer:
        config_dreamer = config_dreamer['dreamer']
        print(f"[DEBUG] Using nested 'dreamer' config")
    
    # Override with our settings - properly update nested dicts
    config_dreamer['logdir'] = dreamer_dir
    
    # Update env config
    print(f"\n[DEBUG] Before update - env.path: {config_dreamer['env']['path']}")
    config_dreamer['env']['path'] = ENV_PATH
    config_dreamer['env']['vision'] = False
    print(f"[DEBUG] After update - env.path: {config_dreamer['env']['path']}")
    
    # Update run config
    print(f"[DEBUG] Before update - run.steps: {config_dreamer['run']['steps']}")
    config_dreamer['run']['steps'] = TRAIN_STEPS
    config_dreamer['run']['eval_every'] = 10000
    config_dreamer['run']['eval_eps'] = 1
    config_dreamer['run']['train_ratio'] = 512
    config_dreamer['run']['from_checkpoint'] = ''
    config_dreamer['run']['num_envs'] = 1  # Fix: Use single env to avoid multiprocessing issues
    config_dreamer['run']['driver_parallel'] = False  # Fix: Disable parallel to avoid agent duplication
    config_dreamer['run']['log_every'] = 30  # Log every 30 seconds for more frequent updates
    print(f"[DEBUG] After update - run.steps: {config_dreamer['run']['steps']}")
    
    # Update jax config
    config_dreamer['jax']['platform'] = 'gpu'  # Use GPU now that we fixed the multiprocessing bug
    
    print(f"\n[DEBUG] Final config type: {type(config_dreamer)}")
    print(f"[DEBUG] Final env.path: {config_dreamer['env']['path']}")
    print(f"[DEBUG] Final run.steps: {config_dreamer['run']['steps']}")

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
    'env_path': ENV_PATH,
    'dreamer_dir': dreamer_dir,
    'train_steps': TRAIN_STEPS,
}

with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"  - Environment: {ENV_PATH}")
print(f"  - Training logs: {dreamer_dir}")
print(f"  - Trained agent GIF: {os.path.join(OUTPUT_DIR, 'trained_agent.gif')}")
print("\nYou can now view the GIF to see how the trained agent performs!")
