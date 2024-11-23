from textwrap import dedent
import numpy as np
import jax
import jax.numpy as jnp

from omni_epic.envs.base import EnvState


robot_dict = {
	 "r2d2": {
		"robot_desc": dedent("""
			R2D2 robot that can fly with thursters and measures 1.0 m in width and 1.5 m in height.
			""").strip(),
		"env_paths_example": [
			"/workdir/omni_epic/envs/wall_is_lava.py",
			"/workdir/omni_epic/envs/avoid_flying_boxes.py",
			"/workdir/omni_epic/envs/destroy_tower.py",
			"/workdir/omni_epic/envs/open_door.py",
		],
		"task_descs_init": [
			dedent("""
			Touch the green block.

			Description:
			- The robot is initialized mid-air in a room
			- There is a green block placed at a random location in the room
			- The robot needs to navigate to and touch the green block

			Success:
			The task is completed if the robot successfully touches the green block.

			Rewards:
			The robot is rewarded for getting closer to the green block.

			Termination:
			The task terminates if the robot touches the green block.
			""").strip(),
			dedent("""
			Touch the green block while avoiding the red blocks.

			Description:
			- The robot is initialized mid-air in a room
			- There is a green block placed at a random location in the room
			- There are 3 red blocks scattered around the room
			- The robot needs to navigate to and touch the green block while avoiding the red blocks

			Success:
			The task is completed if the robot successfully touches the green block.

			Rewards:
			The robot gets a bonus of 100 points if it successfully touches the green block.
			The robot is penalized by 100 points for touching any red block.

			Termination:
			The task terminates if the robot successfully touches the green block.
			""").strip(),
		]
	}
}

# Test Environment
terminated_error = """
The method Env.get_terminated returns True immediately following Env.reset, leading the episode to terminate prematurely.

Possible causes:
- The method Env.get_terminated might not be implemented correctly.
- The method Env.reset might not be implemented correctly, causing the termination condition to be met immediately after reset.

To fix:
- Check the implementation of Env.get_terminated and ensure that the logic is correct.
- Check the implementation of Env.reset and make sure that the termination condition is not met immediately after reset. For example, ensure that the initial state of the robot does not meet the termination condition after reset.
""".strip()

success_error = """
The method Env.get_success returns True immediately following Env.reset, leading to completing the task prematurely.

Possible causes:
- The method Env.get_success might not be implemented correctly.
- The method Env.reset might not be implemented correctly, causing the success condition to be met immediately after reset.

To fix:
- Check the implementation of Env.get_success and ensure that the logic is correct.
- Check the implementation of Env.reset and make sure that the success condition is not met immediately after reset.
	- Ensure that the initial state of the robot does not meet the success condition after reset.
""".strip()

robot_colliding_error = """
A collision has been detected between the robot and another body, immediately following Env.reset. This issue typically indicates a problem with the initial position of the robot relative to its environment, leading to overlaps.

Possible causes:
- The initial position of the robot might be set incorrectly during Env.reset.
- The initial position or orientation of at least one object might be set incorrectly during Env.reset.

To fix:
- Ensure that the robot's initial position is set relative to the platform it starts on, as demonstrated in the provided environment code examples. For example, if the robot starts on a platform, its initial position should be set to [self.platform_position[0], self.platform_position[1], self.platform_position[2] + self.platform_size[2] / 2 + self.robot.links["base"].position_init[2]].
- Check Env.reset and make sure that the initial position of the robot is set correctly.
	- Ensure that the initial x and y coordinates of the robot are set to the designated starting point of the supporting ground or platform to avoid off-edge placements.
	- Ensure that the initial z coordinate of the robot is set to a height that allows for safe clearance above the supporting ground or platform, avoiding any unintended collision with the surface.
- Check Env.reset and make sure that the initial position of the objects are set correctly.
	- Ensure that the initial position of each object is spaced far enough from the robot, taking into account the size and shape of each object to prevent overlapping.
	- Ensure that the initial orientation of each object is appropriate, and that any directional aspects of the objects do not interfere with the robot's starting position.
""".strip()

object_colliding_error = """
A collision has been detected between at least two bodies, immediately following Env.reset. This issue typically indicates a problem with the initial position or orientation of the different bodies, leading to overlaps.

To fix:
- Check Env.reset and make sure that the initial position and orientation of each object are set correctly.
	- If an object is supposed to be initialized on a supporting ground or platform, ensure that the initial x and y coordinates of the object are set to the designated starting point of the supporting ground or platform to avoid off-edge placements.
	- If an object is supposed to be initialized on a supporting ground or platform, ensure that the initial z coordinate of the object is set to a height that allows for safe clearance above the supporting ground or platform, avoiding any unintended collision with the surface.
- Ensure that objects are spaced far enough from each other, taking into account the size and shape of each object to prevent overlapping.
- Ensure that the initial orientation of each object is appropriate, and that any directional aspects of objects do not interfere with each other.
""".strip()

robot_falling_error = """
The robot is falling immediately following Env.reset.

Possible causes:
- The initial position of the robot might be set incorrectly during Env.reset, causing it to start off the edge of a platform or unsupported area.
- No supporting ground or platform for the robot to stand on has been created during Env.reset, causing the robot to free fall.
- A supporting ground of platform for the robot to stand on exists, but it might not be large enough or its initial position might be set incorrectly, leading to inadequate support.

To fix:
- Check Env.reset and make sure that the initial position of the robot is set correctly.
	- Verify that the robot is initialized at a safe and central position on the platform or ground. Check the x and y coordinates to ensure they center the robot adequately on the available space.
	- Ensure the z coordinate positions the robot firmly on the surface, without any part suspended in air.
- Confirm the existence and adequacy of the platform or ground:
	- Check that a platform or ground is created to support the robot.
	- Ensure that the platform or ground is of appropriate dimensions to accommodate the robot's size.
	- Adjust the initial position of the platform or ground, making sure it aligns correctly with the initial position of the robot.
	- Make sure that the platform or ground is steady and stable, providing a secure foundation for the robot.
""".strip()

timeout_error = """
A method in class Env exceeded the time limit while running.

Possible causes:
- A method might contain an infinite loop.
- A method might take an excessive amount of time to complete.

To fix:
Check the implementation of Env and ensure that all methods including Env.__init__ have proper termination conditions and don't contain infinite loops.
""".strip()


class EnvironmentError(Exception):
	pass


def test_env(env_path):
	# Test Env.__init__
	from ppo.wrappers import Jax2DWrapper

	env = Jax2DWrapper(env_path)

	try:
		# Test Env.reset
		key = jax.random.PRNGKey(0)
		env_state = env.reset(key)
		if not isinstance(env_state, EnvState):
			raise EnvironmentError(
				f"Expected observation from Env.reset to be a numpy.ndarray, but received type '{type(env_state).__name__}'. "
				"Please ensure that observation from Env.reset returns a numpy.ndarray."
			)

		# # Test robot collision after Env.reset
		# if env.is_robot_colliding():
		# 	raise EnvironmentError(robot_colliding_error)

		# Test Env.step
		action = 0.0 * env.action_space.sample()
		env_state = env.step(key, env_state, action)

		if not isinstance(env_state, EnvState):
			raise EnvironmentError(
				f"Expected observation from Env.step to be a numpy.ndarray, but received type '{type(env_state).__name__}'. "
				"Please ensure that observation from Env.step returns a numpy.ndarray."
			)

		if not (isinstance(env_state.terminated, jax.Array) and (env_state.terminated.dtype == jnp.float32 or env_state.terminated.dtype == jnp.bool_)):
			raise EnvironmentError(
				f"Expected terminated from Env.step to be a boolean, but received type '{type(env_state.terminated).__name__}'. "
				"Please ensure that terminated from Env.step returns a boolean."
			)

		# Test Env.get_success
		manifolds = env.get_manifolds(env_state)
		success = env.get_success(env_state, manifolds, action)
		if not (isinstance(success, jax.Array) and (success.dtype == jnp.float32 or success.dtype == jnp.bool_)):
			raise EnvironmentError(
				f"Expected success from Env.get_success to be a boolean, but received type '{type(success).__name__}'. "
				"Please ensure that success from Env.get_success returns a boolean."
			)

		# # Test robot collision after one Env.step call
		# if env.is_robot_colliding():
		# 	raise EnvironmentError(robot_colliding_error)

		# Test terminated after one Env.step call
		if env_state.terminated:
			raise EnvironmentError(terminated_error)

		# Test success after one Env.step call
		if success:
			raise EnvironmentError(success_error)

		for _ in range(100):
			env.step(key, env_state, 0.0 * env.action_space.sample())
	except Exception as e:
		raise e
	finally:
		env.close()


if __name__ == "__main__":
	env_path = "/workspace/src/env_not_halting.py"
	# env_path = "/workspace/src/env_error.py"
	# env_path = "/workspace/src/env_good.py"

