from functools import partial

import jax
import jax.numpy as jnp

from omni_epic.envs.base import EnvBase


class Env(EnvBase):
	"""
	Push button to open door and touch the flag.

	Description:
	- The robot is initialized mid-air in a room
	- There is a door that can be opened by pressing a button
	- There is a flag behind the door

	Success:
	The task is completed if the robot successfully touches the flag.

	Rewards:
	Before opening the door, the robot is rewarded for getting closer to the button.
	After opening the door, the robot is rewarded for getting closer to the flag.

	Termination:
	The task terminates if the robot touches the flag.
	"""

	def __init__(self):
		super().__init__()

		# Add button
		self.env_state_init, self.button_idx = self.add_circle_to_scene(
			self.env_state_init,
			position=jnp.array([4 * self.scene_size / 5, self.scene_size / 5]),
			radius=1.0,
			fixated=True,
			color=jnp.array([0.0, 0.0, 1.0]),
		)

		# Add door
		self.env_state_init, self.door_idx = self.add_rectangle_to_scene(
			self.env_state_init,
			position=jnp.array([self.scene_size / 2, 3 * self.scene_size / 5]),
			dimensions=jnp.array([self.scene_size, 1.0]),
			fixated=True,
			color=jnp.array([0.0, 0.0, 0.0]),
		)

		# Add flag behind door
		self.env_state_init, self.flag_idx = self.add_circle_to_scene(
			self.env_state_init,
			position=jnp.array([self.scene_size / 2, 4 * self.scene_size / 5]),
			radius=1.0,
			fixated=True,
			color=jnp.array([0.0, 1.0, 0.0]),
		)

	@partial(jax.jit, static_argnames=("self",))
	def reset(self, key):
		# Set initial position of the robot
		position = jax.random.uniform(key, (2,), minval=self.scene_size / 5, maxval=2 * self.scene_size / 5)
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

		# Check if robot touches the button
		button_collision = self.collision_cp(
			self.get_manifolds(env_state),
			self.robot_idx,
			self.button_idx,
		)

		# Remove door if button is pressed
		env_state = jax.lax.cond(
			button_collision,
			lambda env_state: self.remove_polygon(env_state, self.door_idx),
			lambda env_state: env_state,
			env_state
		)

		return env_state

	@partial(jax.jit, static_argnames=("self",))
	def get_task_rewards(self, env_state, manifolds, action):
		distance_to_button = self.dist_cp(env_state, self.button_idx, self.robot_idx)
		distance_to_flag = self.dist_cp(env_state, self.flag_idx, self.robot_idx)
		door_removed = self.dist_pp(env_state, self.robot_idx, self.door_idx) > 1e3  # The remove method puts the object far away

		return {
			"distance_to_button": jnp.where(door_removed, 0.0, -distance_to_button),
			"distance_to_flag": jnp.where(door_removed, -distance_to_flag, 0.0)
		}

	@partial(jax.jit, static_argnames=("self",))
	def get_terminated(self, env_state, manifolds, action):
		return jnp.array(0.0)

	@partial(jax.jit, static_argnames=("self",))
	def get_success(self, env_state, manifolds, action):
		return self.collision_cp(manifolds, self.robot_idx, self.flag_idx)
