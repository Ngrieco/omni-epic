#!/usr/bin/env python3
"""
Render a trained agent from a checkpoint.

Usage:
    python render_from_checkpoint.py --checkpoint path/to/checkpoint.ckpt --env path/to/env.py
    
    Or let it auto-detect from checkpoint directory:
    python render_from_checkpoint.py --checkpoint path/to/checkpoint.ckpt
"""

import os
import sys
import dreamerv3
import embodied
from pathlib import Path

# Parse arguments
if len(sys.argv) < 3 or sys.argv[1] != '--checkpoint':
    print("Usage: python render_from_checkpoint.py --checkpoint path/to/checkpoint.ckpt [--env path/to/env.py]")
    sys.exit(1)

CHECKPOINT_PATH = sys.argv[2]

if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: Checkpoint not found: {CHECKPOINT_PATH}")
    sys.exit(1)

# Check for optional --env argument
env_path = None
if len(sys.argv) >= 5 and sys.argv[3] == '--env':
    env_path = sys.argv[4]
    if not os.path.exists(env_path):
        print(f"Error: Environment file not found: {env_path}")
        sys.exit(1)

print("="*80)
print("RENDER FROM CHECKPOINT")
print("="*80)
print(f"\nCheckpoint: {CHECKPOINT_PATH}")

# Extract info from checkpoint path
checkpoint_dir = Path(CHECKPOINT_PATH).parent
logdir = checkpoint_dir.parent

print(f"Log directory: {logdir}")

# If env_path not provided, try to auto-detect
if not env_path:
    # Find the environment path from the checkpoint directory
    # Look for metrics.jsonl which should contain env info
    metrics_file = logdir / 'metrics.jsonl'
    if not metrics_file.exists():
        print(f"Error: Could not find metrics.jsonl in {logdir}")
        print("Cannot determine which environment was used for training")
        print("Please specify the environment manually with --env")
        sys.exit(1)

    # Try to find env path from config or metrics
    import json

    # Check if there's a config file
    config_file = logdir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            if 'env' in config_data and 'path' in config_data['env']:
                env_path = config_data['env']['path']

    if not env_path:
        print("Error: Could not determine environment path from checkpoint")
        print("Please specify the environment manually with --env")
        sys.exit(1)

print(f"Environment: {env_path}")

# Create config for rendering
from dreamerv3 import agent as agt
config = embodied.Config(agt.Agent.configs['defaults'])

# Add env config with path
env_config = {
    'path': env_path,
    'vision': False,
    'size': (64, 64),
    'use_depth': False,
    'fov': 90,
}
config._nested['env'].update(env_config)

config = config.update({
    'logdir': str(logdir),
    'jax.policy_devices': [0],
    'jax.train_devices': [0],
})
config, _ = embodied.Flags(config).parse_known()

print("\n[RENDER] Creating environment...")
from embodied.envs.pybullet import PyBullet
env = PyBullet(config.env.path, vision=config.env.vision, size=config.env.size, 
               use_depth=config.env.use_depth, fov=config.env.fov)
env = dreamerv3.wrap_env(env, config)

print("[RENDER] Creating agent...")
agent = dreamerv3.Agent(env.obs_space, env.act_space, config)

print(f"[RENDER] Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = embodied.Checkpoint()
checkpoint.load(CHECKPOINT_PATH, keys=['agent'])
agent.load(checkpoint['agent'])

print("[RENDER] Running episode...")
obs = env.step({'action': env.act_space['action'].sample(), 'reset': True})

frames_1p = []
frames_3p = []
total_reward = 0
step_count = 0

while not obs['is_last']:
    # Get action from agent
    action = agent.policy(obs, mode='eval')
    
    # Step environment
    obs = env.step(action)
    total_reward += obs['reward']
    step_count += 1
    
    # Render
    frame_1p = env.render(height=480, width=640)
    frame_3p = env.render3p(height=480, width=640)
    
    frames_1p.append(frame_1p)
    frames_3p.append(frame_3p)
    
    if step_count % 50 == 0:
        print(f"  Step {step_count} - Reward: {total_reward:.3f}")

print(f"\n[RENDER] Episode complete!")
print(f"  Total steps: {step_count}")
print(f"  Total reward: {total_reward:.3f}")
print(f"  Success: {env.get_success()}")

# Save as GIFs
print("\n[RENDER] Saving GIFs...")
from PIL import Image

output_dir = logdir / 'render_output'
output_dir.mkdir(exist_ok=True)

# First-person view
images_1p = [Image.fromarray(frame) for frame in frames_1p]
gif_path_1p = output_dir / 'agent_firstperson.gif'
images_1p[0].save(
    gif_path_1p,
    save_all=True,
    append_images=images_1p[1:],
    duration=50,
    loop=0
)
print(f"[RENDER] ✓ First-person GIF: {gif_path_1p}")

# Third-person view
images_3p = [Image.fromarray(frame) for frame in frames_3p]
gif_path_3p = output_dir / 'agent_thirdperson.gif'
images_3p[0].save(
    gif_path_3p,
    save_all=True,
    append_images=images_3p[1:],
    duration=50,
    loop=0
)
print(f"[RENDER] ✓ Third-person GIF: {gif_path_3p}")

env.close()

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nGIFs saved to: {output_dir}")
