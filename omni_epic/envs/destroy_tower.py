from functools import partial

import jax
import jax.numpy as jnp

from omni_epic.envs.base import EnvBase


class Env(EnvBase):
	"""
	Destroy the tower of stacked boxes.

	Description:
	- The robot is initialized mid-air in an empty room with walls, ceiling and floor
	- There is a tower of 5 stacked boxes on the right side of the room
	- The boxes are colored in a gradient from blue to red
	The task of the robot is to destroy the tower of boxes by knocking them over.

	Success:
	The task is completed when the top box in the tower has been knocked over.

	Rewards:
	The robot receives a penalty at each time step proportional to the distance to the top box in the tower.

	Termination:
	The task terminates if the top box in the tower has been knocked over and touches the floor.
	"""

	def __init__(self):
		super().__init__()

		self.block_indices = []
		base_x = 10.0
		block_size = jnp.array([2.0, 2.0])

		for i in range(5):
			# Calculate position - each block stacks on top of previous
			block_position = jnp.array([base_x, 1.0 + i * 2.0])

			# Create color gradient from blue to red
			color = jnp.array([i / 4, 0.0, 1.0 - i / 4])

			self.env_state_init, block_idx = self.add_rectangle_to_scene(
				self.env_state_init,
				position=block_position,
				dimensions=block_size,
				density=0.1,
				color=color,
			)

			self.block_indices.append(block_idx)

	@partial(jax.jit, static_argnames=("self",))
	def reset(self, key):
		# Set initial position of the robot
		position = jax.random.uniform(key, (2,), minval=2.0, maxval=4.0)
		env_state = self.set_polygon_position(self.env_state_init, self.robot_idx, position)

		env_state = super().reset(env_state)

		return env_state

	@partial(jax.jit, static_argnames=("self",))
	def step(self, key, env_state, action):
		actions = jnp.zeros(
			self.static_sim_params.num_joints + self.static_sim_params.num_thrusters
		)

		actions = self.apply_action(actions, action)
		env_state = super().step(env_state, actions)

		return env_state

	@partial(jax.jit, static_argnames=("self",))
	def get_task_rewards(self, env_state, manifolds, action):
		# Get a penalty proportional to the distance as long as the tower is still standing
		distance_to_block = -self.dist_pp(env_state, self.robot_idx, self.block_indices[-1])
		return {"reward": distance_to_block}

	@partial(jax.jit, static_argnames=("self",))
	def get_terminated(self, env_state, manifolds, action):
		return self.collision_pp(manifolds, self.floor_idx, self.block_indices[-1])

	@partial(jax.jit, static_argnames=("self",))
	def get_success(self, env_state, manifolds, action):
		return self.collision_pp(manifolds, self.floor_idx, self.block_indices[-1])
