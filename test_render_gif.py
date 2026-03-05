#!/usr/bin/env python3
"""Test script to render a random episode and save as GIF."""

import numpy as np
from embodied.envs.pybullet import PyBullet
from PIL import Image
import os

print('[RENDER TEST] Creating environment...')
env_wrapper = PyBullet(env_path='omni_epic/envs/r2d2/cross_bridge.py', vision=False)
env = env_wrapper._env

print('[RENDER TEST] Resetting environment...')
env.reset()

print('[RENDER TEST] Running random episode and collecting frames...')
frames = []
frames3p = []

# Get render config for optimal camera positioning
render_config = env.get_render_config()
print(f'[RENDER TEST] Render config: {render_config}')

# Collect frames for 200 steps
num_steps = 200
for step in range(num_steps):
    if step % 50 == 0:
        print(f'[RENDER TEST] Step {step}/{num_steps}...')
    
    # Render both views
    frame = env.render(height=480, width=640, **render_config)
    frame3p = env.render3p(height=480, width=640)
    
    frames.append(frame)
    frames3p.append(frame3p)
    
    # Take random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f'[RENDER TEST] Episode ended at step {step}')
        break

env.close()

print(f'[RENDER TEST] Collected {len(frames)} frames')

# Save as GIF
output_dir = '/test_renders'
os.makedirs(output_dir, exist_ok=True)

print('[RENDER TEST] Saving main view GIF...')
images = [Image.fromarray(frame) for frame in frames]
gif_path = os.path.join(output_dir, 'test_render_main.gif')
images[0].save(
    gif_path,
    save_all=True,
    append_images=images[1:],
    duration=50,  # 50ms per frame = 20 FPS
    loop=0
)
print(f'[RENDER TEST] ✓ Saved main view to: {gif_path}')

print('[RENDER TEST] Saving third-person view GIF...')
images3p = [Image.fromarray(frame) for frame in frames3p]
gif_path3p = os.path.join(output_dir, 'test_render_3rdperson.gif')
images3p[0].save(
    gif_path3p,
    save_all=True,
    append_images=images3p[1:],
    duration=50,
    loop=0
)
print(f'[RENDER TEST] ✓ Saved third-person view to: {gif_path3p}')

print('[RENDER TEST] Done! Check the GIFs to see how rendering looks.')
