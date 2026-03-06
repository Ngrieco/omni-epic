# Scenario-Based Training System

This system allows you to define training scenarios in simple markdown files and automatically train agents on them.

## Quick Start

### 1. Create a Scenario File

Create a markdown file in the `scenarios/` directory:

```markdown
# My Custom Scenario

## Task Description
Your task description here (e.g., "Navigate through a maze")

## Environment Settings
- **Robot**: r2d2
- **Vision**: false
- **Environment Size**: [10, 10, 10]

## Training Settings
- **Steps**: 100000
- **Batch Size**: 16
- **Batch Length**: 64
- **Eval Episodes**: 5
- **Eval Every**: 10000

## Custom Environment Path (Optional)
Leave empty to auto-generate, or specify existing environment:
```
env_path: omni_epic/envs/r2d2/my_env.py
```

## Notes
Additional notes about your scenario.
```

### 2. Run Training

```bash
cd omni-epic
python load_scenario.py scenarios/my_scenario.md
```

## How It Works

1. **Parse Scenario**: Reads your markdown file and extracts configuration
2. **Generate/Load Environment**: 
   - If `env_path` is specified: Uses existing environment
   - If empty: Generates new environment from task description using claude-4-5-sonnet-latest
3. **Train Agent**: Runs Dreamer training with your specified parameters
4. **Save Results**: Outputs to `omni-epic/scenarios/runs/<scenario_name>/`

## Example Scenarios

### Using Existing Environment

```bash
# Train on cross_bridge task
python load_scenario.py scenarios/cross_bridge.md
```

### Generating New Environment

```bash
# Will generate environment from task description
python load_scenario.py scenarios/example_scenario.md
```

## Scenario File Format

### Required Sections

#### Task Description
Natural language description of what the robot should do.

```markdown
## Task Description
Navigate through a maze with moving obstacles.
```

#### Environment Settings
- **Robot**: Robot type (e.g., r2d2, ant, humanoid)
- **Vision**: true/false - whether to use vision
- **Environment Size**: [x, y, z] dimensions

#### Training Settings
- **Steps**: Total training steps
- **Batch Size**: Batch size for training
- **Batch Length**: Sequence length
- **Eval Episodes**: Number of evaluation episodes
- **Eval Every**: Evaluate every N steps

### Optional Sections

#### Custom Environment Path
Specify path to existing environment file, or leave empty to auto-generate.

```markdown
## Custom Environment Path (Optional)
```
env_path: omni_epic/envs/r2d2/custom_env.py
```
```

#### Notes
Any additional information about the scenario.

## Output Structure

```
omni-epic/scenarios/
├── my_scenario.md              # Your scenario file
├── generated/                  # Auto-generated environments
│   └── my_scenario/
│       ├── env_0.py           # Generated environment code
│       ├── render.gif         # Visualization
│       └── metadata.json      # Generation info
└── runs/                      # Training outputs
    └── my_scenario/
        ├── dreamer/           # Dreamer checkpoints
        │   ├── checkpoint.ckpt
        │   ├── metrics.jsonl
        │   └── eval/          # Evaluation episodes
        └── gifs/              # Rendered episodes
```

## Tips

### Quick Testing
For quick testing, reduce training steps:
```markdown
- **Steps**: 10000
- **Eval Every**: 2000
```

### High Quality Training
For production training:
```markdown
- **Steps**: 1000000
- **Batch Size**: 32
- **Eval Every**: 50000
```

### Custom Tasks
Be specific in task descriptions:
- ✅ "Navigate through a maze while avoiding moving red obstacles and collecting blue objects"
- ❌ "Do something interesting"

## Troubleshooting

### Environment Generation Fails
- Simplify task description
- Check OpenAI API key is set
- Try using existing environment instead

### Training Issues
- Reduce batch size if out of memory
- Check environment path is correct
- Verify robot type matches environment

## Advanced Usage

### Multiple Scenarios
Run multiple scenarios in sequence:
```bash
for scenario in scenarios/*.md; do
    python load_scenario.py "$scenario"
done
```

### Custom Monitoring
Monitor training in real-time:
```bash
# Terminal 1: Run training
python load_scenario.py scenarios/my_scenario.md

# Terminal 2: Monitor and create GIFs
python monitor_and_render.py
```

## Examples

See included example scenarios:
- `cross_bridge.md` - Existing environment example
- `example_scenario.md` - Auto-generation example
