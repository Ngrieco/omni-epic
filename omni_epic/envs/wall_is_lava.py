from functools import partial

import jax
import jax.numpy as jnp

from omni_epic.envs.base import EnvBase


class Env(EnvBase):
	"""
	Keep flying in the air without touching the walls, ceiling or floor.

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

	@partial(jax.jit, static_argnames=("self",))
	def reset(self, key):
		# Set initial position of the robot
		position = jax.random.uniform(key, (2,), minval=6.0, maxval=10.0)
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
		return {"reward": 5.0}

	@partial(jax.jit, static_argnames=("self",))
	def get_terminated(self, env_state, manifolds, action):
		return (
			self.collision_pp(manifolds, self.robot_idx, self.ceiling_idx)
			| self.collision_pp(manifolds, self.robot_idx, self.floor_idx)
			| self.collision_pp(manifolds, self.robot_idx, self.right_wall_idx)
			| self.collision_pp(manifolds, self.robot_idx, self.left_wall_idx)
		)

	@partial(jax.jit, static_argnames=("self",))
	def get_success(self, env_state, manifolds, action):
		return env_state.step > 200 & ~self.collision_pp(
			manifolds, self.robot_idx, self.ceiling_idx
		) & ~self.collision_pp(manifolds, self.robot_idx, self.floor_idx) & ~self.collision_pp(
			manifolds, self.robot_idx, self.right_wall_idx
		) & ~self.collision_pp(manifolds, self.robot_idx, self.left_wall_idx)
