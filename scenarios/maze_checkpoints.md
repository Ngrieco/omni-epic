# Maze with Checkpoints

## Task Description
Maze solver agent

Description:
Navigate through a maze to reach the exit at the end. The maze has multiple checkpoints along the path, and the agent receives rewards for reaching each checkpoint, with increasing rewards as it gets closer to the final exit. The agent must learn the optimal path through the maze while avoiding dead ends. The checkpoints should be little golden starts or spheres.
- A player start inside of a maze thats 20x20 m in size.
- The walls are 5 m talls and the player cannot go through the walls or over them.
- The paths are 3m wide for space to navigate
- The maze has only one exit and several dead end paths
- The task of the player is the escape the maze as fast as possible.

Success:
The task is successfully completed when the robot reaches the maze exit.

Rewards:
To help the robot complete the task:
- The robot receives a reward for reaching a checkpoint which indicates the agent is going in the correct direction
- The robot is rewarded based on collecting checkpoints and escaping the maze.
- Small reward when the agent explores.

Termination:
The task terminates only after a specified number of timesteps selected that allows the agent to explore and solve the maze. 