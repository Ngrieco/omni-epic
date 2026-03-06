#!/usr/bin/env python3
"""
Generate a custom task environment from a text description using claude-4-5-sonnet-latest.

Usage:
    python generate_custom_task.py "Your task description here"
    
Example:
    python generate_custom_task.py "Navigate through a maze with moving obstacles"
"""

import sys
import os
import json
import hydra
from omegaconf import OmegaConf, DictConfig

from omni_epic.robots import robot_dict
from omni_epic.core.fm import FM
from run_utils import parse_task_desc_from_env_code


def generate_custom_task(task_description: str, robot: str = "r2d2", output_dir: str = "omni-epic/custom_tasks"):
    """
    Generate a custom task environment from a text description.
    
    Args:
        task_description: Natural language description of the task
        robot: Robot type to use (default: "r2d2")
        output_dir: Directory to save generated environment
    """
    
    print("="*80)
    print("CUSTOM TASK GENERATION")
    print("="*80)
    print(f"\nTask Description: {task_description}")
    print(f"Robot: {robot}")
    print(f"Output Directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    with hydra.initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
        config = hydra.compose(config_name="omni_epic")
    
    # Set up environment generator
    config_env_generator = config.environment_generator
    fm_env_generator = FM(config_env_generator)
    
    print("\n[GENERATE] Querying claude-4-5-sonnet-latest to generate environment code...")
    
    # Generate environment code from task description
    taskgen_completion = fm_env_generator.query_env_code(
        robot, 
        task_description,
        add_examples=config.add_examples
    )
    
    print("[GENERATE] Initial code generated, checking for compilation errors...")
    
    # Iterate on compilation errors
    gen_num = fm_env_generator.iterate_on_errors(
        robot,
        task_description,
        taskgen_completion,
        output_dir,
        add_examples=config.add_examples,
        env_paths_other=[],
        iteration_max=config.error_max_iterations,
    )
    
    if gen_num >= 0:
        print(f"\n[GENERATE] ✓ Environment code generated successfully after {gen_num} iterations!")
        
        # Get the generated environment path
        env_path = os.path.abspath(os.path.join(output_dir, f'env_{gen_num}.py'))
        
        print(f"\n[GENERATE] Environment saved to: {env_path}")
        
        # Save metadata
        metadata = {
            'task_description': task_description,
            'robot': robot,
            'env_path': env_path,
            'generation_iterations': gen_num,
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"[GENERATE] Metadata saved to: {metadata_path}")
        
        # Print next steps
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Train an agent on this environment:")
        print(f"   cd omni-epic")
        print(f"   python simple_train_and_render.py")
        print(f"   # (Update ENV_PATH in the script to: {env_path})")
        
        print("\n2. Or use the full pipeline:")
        print(f"   python main_omni_epic.py override_vars.task_description='{task_description}'")
        
        return env_path
    else:
        print(f"\n[GENERATE] ✗ Failed to generate valid environment code")
        print(f"[GENERATE] Check {output_dir} for error logs")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_custom_task.py \"Your task description here\"")
        print("\nExamples:")
        print('  python generate_custom_task.py "Navigate through a maze with moving obstacles"')
        print('  python generate_custom_task.py "Collect colored objects in a specific order"')
        print('  python generate_custom_task.py "Balance on a narrow beam while avoiding falling objects"')
        sys.exit(1)
    
    task_description = sys.argv[1]
    
    # Optional: specify robot type
    robot = sys.argv[2] if len(sys.argv) > 2 else "r2d2"
    
    # Optional: specify output directory
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "omni-epic/custom_tasks"
    
    env_path = generate_custom_task(task_description, robot, output_dir)
    
    if env_path:
        print("\n✓ Custom task generation complete!")
    else:
        print("\n✗ Custom task generation failed!")
        sys.exit(1)
