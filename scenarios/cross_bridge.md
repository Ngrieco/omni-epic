# Cross Bridge Scenario

## Task Description
Navigate across a narrow bridge to reach the goal on the other side.

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
Using existing cross_bridge environment:
```
env_path: omni_epic/envs/r2d2/cross_bridge.py
```

## Notes
This is the existing cross_bridge task. The robot must carefully navigate across
a narrow bridge without falling off. This tests balance and precise movement control.
