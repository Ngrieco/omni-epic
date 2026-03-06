# Kick blue ball

## Task Description
Kick blue ball

Description:
An agent runs and kicks a blue ball repeatedly to maximize the balls travel distance
- A player start at the center of a 20x20 platform with a blue ball randomly place on the circle with radius 5 around him
- The agent must run to the ball and hit it
- If he hits the ball it bounces away from him
- The task of the player is to hit the ball as far as he can given 10 hits

Success:
The task is successfully completed when the robot hits the ball 10 times

Rewards:
To help the robot complete the task:
- The robot receives a small reward for getting closer to the ball
- The robot gets a large reward proportional to how far he hits it
- A penalty if the agent or the ball go off the platform

Termination:
The task terminates if the agent or the ball go off the platform, the ball is hit 10 times or time runs out.