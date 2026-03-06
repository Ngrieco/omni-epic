#!/usr/bin/env python3
"""
Load scenario configuration from markdown file and run training.

Usage:
    python load_scenario.py scenarios/example_scenario.md
"""

import sys
import os
import re
from pathlib import Path


def parse_scenario_md(md_path):
    """Parse scenario markdown file and extract configuration."""
    
    with open(md_path, 'r') as f:
        content = f.read()
    
    config = {
        'task_description': '',
        'robot': 'r2d2',
        'vision': True,
        'env_size': [10, 10, 10],
        'steps': 100000,
        'batch_size': 16,
        'batch_length': 64,
        'eval_episodes': 5,
        'eval_every': 10000,
        'env_path': None,
        'notes': ''
    }
    
    # Extract task description
    task_match = re.search(r'## Task Description\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if task_match:
        config['task_description'] = task_match.group(1).strip()
    
    # Extract environment settings
    robot_match = re.search(r'\*\*Robot\*\*:\s*(\w+)', content)
    if robot_match:
        config['robot'] = robot_match.group(1)
    
    vision_match = re.search(r'\*\*Vision\*\*:\s*(\w+)', content)
    if vision_match:
        config['vision'] = vision_match.group(1).lower() == 'true'
    
    size_match = re.search(r'\*\*Environment Size\*\*:\s*\[([\d,\s]+)\]', content)
    if size_match:
        config['env_size'] = [int(x.strip()) for x in size_match.group(1).split(',')]
    
    # Extract training settings
    steps_match = re.search(r'\*\*Steps\*\*:\s*(\d+)', content)
    if steps_match:
        config['steps'] = int(steps_match.group(1))
    
    batch_size_match = re.search(r'\*\*Batch Size\*\*:\s*(\d+)', content)
    if batch_size_match:
        config['batch_size'] = int(batch_size_match.group(1))
    
    batch_length_match = re.search(r'\*\*Batch Length\*\*:\s*(\d+)', content)
    if batch_length_match:
        config['batch_length'] = int(batch_length_match.group(1))
    
    eval_eps_match = re.search(r'\*\*Eval Episodes\*\*:\s*(\d+)', content)
    if eval_eps_match:
        config['eval_episodes'] = int(eval_eps_match.group(1))
    
    eval_every_match = re.search(r'\*\*Eval Every\*\*:\s*(\d+)', content)
    if eval_every_match:
        config['eval_every'] = int(eval_every_match.group(1))
    
    # Extract custom environment path
    env_path_match = re.search(r'env_path:\s*([^\s#]+)', content)
    if env_path_match:
        config['env_path'] = env_path_match.group(1).strip()
    
    # Extract notes
    notes_match = re.search(r'## Notes\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if notes_match:
        config['notes'] = notes_match.group(1).strip()
    
    return config


def run_training_from_scenario(scenario_path):
    """Load scenario and run training."""
    
    print("="*80)
    print("SCENARIO-BASED TRAINING")
    print("="*80)
    print(f"\nLoading scenario from: {scenario_path}")
    
    # Parse scenario
    config = parse_scenario_md(scenario_path)
    
    print("\n[CONFIG] Scenario Configuration:")
    print(f"[CONFIG]   Task: {config['task_description'][:80]}...")
    print(f"[CONFIG]   Robot: {config['robot']}")
    print(f"[CONFIG]   Vision: {config['vision']}")
    print(f"[CONFIG]   Steps: {config['steps']}")
    print(f"[CONFIG]   Batch Size: {config['batch_size']}")
    print(f"[CONFIG]   Eval Every: {config['eval_every']}")
    
    # Determine environment path
    if config['env_path']:
        print(f"\n[ENV] Using existing environment: {config['env_path']}")
        env_path = config['env_path']
    else:
        print(f"\n[ENV] Generating environment from task description...")
        from omni_epic.core.fm import FM
        from hydra import compose, initialize_config_dir
        
        # Load config
        with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
            omni_config = compose(config_name="omni_epic")
        
        # Generate environment
        fm = FM(omni_config.environment_generator)
        
        # Create output directory
        scenario_name = Path(scenario_path).stem
        output_dir = f"omni-epic/scenarios/generated/{scenario_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate environment code
        completion = fm.query_env_code(
            config['robot'],
            config['task_description'],
            add_examples=omni_config.add_examples
        )
        
        gen_num = fm.iterate_on_errors(
            config['robot'],
            config['task_description'],
            completion,
            output_dir,
            add_examples=omni_config.add_examples,
            env_paths_other=[],
            iteration_max=omni_config.error_max_iterations,
        )
        
        if gen_num < 0:
            print("[ERROR] Failed to generate environment")
            return False
        
        env_path = os.path.abspath(os.path.join(output_dir, f'env_{gen_num}.py'))
        print(f"[ENV] Generated environment: {env_path}")
    
    # Run training
    print(f"\n[TRAIN] Starting training...")
    
    from main_dreamer import main_dreamer
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    
    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
        dreamer_config = compose(config_name="dreamer/dreamer_xxs")
        
        # Update config with scenario settings
        OmegaConf.set_struct(dreamer_config, False)
        
        if 'dreamer' in dreamer_config:
            dreamer_config = dreamer_config['dreamer']
        
        # Set paths
        scenario_name = Path(scenario_path).stem
        dreamer_config['logdir'] = f"omni-epic/scenarios/runs/{scenario_name}"
        dreamer_config['env']['path'] = env_path
        dreamer_config['env']['vision'] = config['vision']
        
        # Set training parameters
        dreamer_config['run']['steps'] = config['steps']
        dreamer_config['batch_size'] = config['batch_size']
        dreamer_config['batch_length'] = config['batch_length']
        dreamer_config['run']['eval_eps'] = config['eval_episodes']
        dreamer_config['run']['eval_every'] = config['eval_every']
        
        print(f"[TRAIN] Log directory: {dreamer_config['logdir']}")
        print(f"[TRAIN] Training for {config['steps']} steps")
        
        # Run Dreamer
        main_dreamer(dreamer_config)
    
    print("\n[TRAIN] Training complete!")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_scenario.py <scenario.md>")
        print("\nExample:")
        print("  python load_scenario.py scenarios/example_scenario.md")
        print("\nAvailable scenarios:")
        scenario_dir = Path("omni-epic/scenarios")
        if scenario_dir.exists():
            for md_file in scenario_dir.glob("*.md"):
                print(f"  - {md_file}")
        sys.exit(1)
    
    scenario_path = sys.argv[1]
    
    if not os.path.exists(scenario_path):
        print(f"Error: Scenario file not found: {scenario_path}")
        sys.exit(1)
    
    success = run_training_from_scenario(scenario_path)
    
    if success:
        print("\n✓ Scenario training completed successfully!")
    else:
        print("\n✗ Scenario training failed!")
        sys.exit(1)
