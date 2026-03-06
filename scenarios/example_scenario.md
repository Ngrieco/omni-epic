# Scenario Configuration

## Task Description
Navigate through a maze with moving obstacles while collecting colored objects.

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
Leave empty to generate from task description, or specify path to existing environment:
```
# env_path: omni_epic/envs/r2d2/custom_maze.py
```

## Notes
This scenario will train an agent to navigate a complex maze environment.
The agent needs to avoid moving obstacles while collecting objects in a specific order.
