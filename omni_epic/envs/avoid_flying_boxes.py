from functools import partial

import jax
import jax.numpy as jnp

from omni_epic.envs.base import EnvBase


class Env(EnvBase):
	"""
	Avoid flying boxes.

	Description:
	- The robot is initialized mid-air in an empty room.
	- The room has walls, a ceiling and a floor.
	The task of the robot is to keep flying in the air without touching the walls, ceiling or floor.

	Success:
	The task is completed if the robot remains flying in the air.

	Rewards:
	The robot is rewarded for remaining flying in the air.

	Termination:
	The task terminates if the robot touches the walls, ceiling or floor.
	"""

	def __init__(self):
		super().__init__()

		# Remove ceiling and floor
		self.env_state_init = self.remove_polygon(self.env_state_init, self.ceiling_idx)
		self.env_state_init = self.remove_polygon(self.env_state_init, self.floor_idx)

		# Add 3 boxes
		self.box_indices = []
		for i in range(3):
			# Create color gradient from blue to red
			color = jnp.array([i / 4, 0.0, 1.0 - i / 4])

			self.env_state_init, box_idx = self.add_rectangle_to_scene(
				self.env_state_init,
				position=jnp.array([5.0 + 3 * i, 16.0]),
				dimensions=jnp.array([2.0, 2.0]),
				density=0.1,
				color=color,
			)
			self.box_indices.append(box_idx)

	@partial(jax.jit, static_argnames=("self",))
	def reset(self, key):
		# Set initial position of the robot
		position = jax.random.uniform(key, (2,), minval=6.0, maxval=10.0)
		env_state = self.set_polygon_position(self.env_state_init, self.robot_idx, position)

		# Set initial position and velocity of the boxes
		for i in range(3):
			# Position
			env_state = self.set_polygon_position(
				env_state, self.box_indices[i], jnp.array([5.0 + 3 * i, 16.0])
			)

			# Velocity
			key, subkey = jax.random.split(key)
			velocity = jax.random.uniform(subkey, (2,), minval=-2.0, maxval=2.0)
			env_state = self.set_polygon_velocity(env_state, self.box_indices[i], velocity)

		env_state = super().reset(env_state)

		return env_state

	@partial(jax.jit, static_argnames=("self",))
	def step(self, key, env_state, action):
		actions = jnp.zeros(
			self.static_sim_params.num_joints + self.static_sim_params.num_thrusters
		)

		actions = self.apply_action(actions, action)
		env_state = super().step(env_state, actions)

		# If box y-position is below the floor, put it above ceiling
		for i in range(3):
			box_position = self.get_polygon_position(env_state, self.box_indices[i])
			box_position = jnp.where(
				box_position[1] < 0.0, jnp.array([box_position[0], 16.0]), box_position
			)
			env_state = self.set_polygon_position(env_state, self.box_indices[i], box_position)

		return env_state

	@partial(jax.jit, static_argnames=("self",))
	def get_task_rewards(self, env_state, manifolds, action):
		# Get a survival reward
		return {"reward": 5.0}

	@partial(jax.jit, static_argnames=("self",))
	def get_terminated(self, env_state, manifolds, action):
		return (
			self.collision_pp(manifolds, self.robot_idx, self.box_indices[0])
			| self.collision_pp(manifolds, self.robot_idx, self.box_indices[1])
			| self.collision_pp(manifolds, self.robot_idx, self.box_indices[2])
			| (self.get_polygon_position(env_state, self.robot_idx)[1] < 0.0)
		)

	@partial(jax.jit, static_argnames=("self",))
	def get_success(self, env_state, manifolds, action):
		return env_state.step > 200 & ~self.collision_pp(
			manifolds, self.robot_idx, self.box_indices[0]
		) & ~self.collision_pp(manifolds, self.robot_idx, self.box_indices[1]) & ~self.collision_pp(
			manifolds, self.robot_idx, self.box_indices[2]
		)
