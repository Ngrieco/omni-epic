from functools import partial, cached_property

import gym.spaces
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax2d.engine import PhysicsEngine
from jax2d.maths import rmat
from jax2d.scene import (
	add_fixed_joint_to_scene,
	add_thruster_to_scene,
)
from jax2d.sim_state import SimParams, StaticSimParams, SimState
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
	add_mask_to_shader,
	fragment_shader_circle,
	make_fragment_shader_convex_dynamic_ngon_with_edges,
)

from omni_epic.jax2d.engine import create_empty_sim
from omni_epic.jax2d.scene import add_circle_to_scene, add_rectangle_to_scene


@struct.dataclass
class EnvState:
	sim_state: SimState
	observation: jax.Array
	reward: float
	terminated: bool
	info: dict


class EnvBase:
	screen_dim = (256, 256)
	scene_size = 16

	action_max = 5.0

	def __init__(self):
		# Create engine with default parameters
		self.static_sim_params = StaticSimParams(
			num_polygons=32,
			num_circles=32,
			num_joints=32,
			num_thrusters=32,
		)
		self.sim_params = SimParams(clip_position=jnp.inf)
		self.physics_engine = PhysicsEngine(self.static_sim_params)
		self.step_fn = jax.jit(self.physics_engine.step)

		# Make renderer
		self.renderer = make_render(self.static_sim_params, self.screen_dim, self.scene_size)

		# Create scene
		sim_state_init = create_empty_sim(
			self.static_sim_params, scene_size=self.scene_size, floor_offset=0.0
		)
		self.floor_idx = 0
		self.left_wall_idx = 1
		self.right_wall_idx = 2
		self.ceiling_idx = 3

		# Initialize environment state
		env_state = EnvState(
			sim_state=sim_state_init,
			observation=jnp.nan,
			reward=jnp.array(0.0),
			terminated=jnp.array(0.0),
			info={},
		)

		# Build robot
		robot_position = jnp.array([8.0, 8.0])
		env_state, self.robot_idx = self.add_rectangle_to_scene(
			env_state,
			position=robot_position,
			dimensions=jnp.array([1.0, 1.0]),
			friction=0.1,
			color=jnp.array([0.75, 0.75, 0.75]),
		)
		env_state, c_idx = self.add_circle_to_scene(
			env_state,
			position=robot_position + jnp.array([0.0, 0.5]),
			radius=0.5,
			density=0.0,
			color=jnp.array([0.1, 0.2, 0.6]),
		)
		env_state, _ = self.add_fixed_joint_to_scene(
			env_state,
			a_index=self.robot_idx,
			b_index=c_idx,
			a_relative_pos=jnp.zeros(2),
			b_relative_pos=jnp.array([0.0, -0.5]),
		)
		env_state, _ = self.add_thruster_to_scene(  # Up thruster
			env_state,
			object_index=self.robot_idx,
			relative_position=jnp.array([0.0, 1.0]),
			rotation=jnp.pi / 2,
		)
		env_state, _ = self.add_thruster_to_scene(  # Right thruster
			env_state,
			object_index=self.robot_idx,
			relative_position=jnp.array([1.0, 0.25]),
			rotation=0.0,
		)
		env_state, _ = self.add_thruster_to_scene(  # Left thruster
			env_state,
			object_index=self.robot_idx,
			relative_position=jnp.array([-1.0, 0.25]),
			rotation=jnp.pi,
		)
		env_state, _ = self.add_thruster_to_scene(  # Rotation thruster
			env_state,
			object_index=self.robot_idx,
			relative_position=jnp.array([0.0, 1.0]),
			rotation=0.0,
		)

		# Initialize robot position
		self.env_state_init = self.set_polygon_position(env_state, self.robot_idx, robot_position)

	@cached_property
	def action_space(self):
		high = np.ones((4,), dtype=np.float32)
		return gym.spaces.Box(-high, high, dtype=np.float32)

	@cached_property
	def observation_space(self):
		return gym.spaces.Dict({
			"vector": gym.spaces.Box(
				low=-np.ones(7, dtype=np.float32),
				high=np.ones(7, dtype=np.float32),
				dtype=np.float32
			),
			"image": gym.spaces.Box(
				low=-np.ones(self.screen_dim + (3,), dtype=np.float32),
				high=np.ones(self.screen_dim + (3,), dtype=np.float32),
				dtype=np.float32
			)
		})

	@partial(jax.jit, static_argnames=("self",))
	def reset(self, env_state):
		actions = jnp.zeros(
			self.static_sim_params.num_joints + self.static_sim_params.num_thrusters
		)

		# Step simulation
		sim_state, manifolds = self.step_fn(env_state.sim_state, self.sim_params, actions)
		env_state = env_state.replace(sim_state=sim_state)

		# Observation
		observation = self.get_observation(env_state, manifolds, actions)

		return env_state.replace(observation=observation)

	@partial(jax.jit, static_argnames=("self",))
	def step(self, env_state, actions):
		# Step simulation
		sim_state, manifolds = self.step_fn(env_state.sim_state, self.sim_params, actions)
		env_state = env_state.replace(sim_state=sim_state)

		# Observation, reward, terminated
		observation = self.get_observation(env_state, manifolds, actions)
		reward = self.get_reward(env_state, manifolds, actions)
		terminated = self.get_terminated(env_state, manifolds, actions)

		return env_state.replace(
			observation=observation,
			reward=reward.astype(jnp.float32),
			terminated=terminated.astype(jnp.float32),
		)

	@partial(jax.jit, static_argnames=("self",))
	def get_reward(self, env_state, manifolds, actions):
		robot_rewards = self.get_robot_rewards(env_state, manifolds, actions)
		task_rewards = self.get_task_rewards(env_state, manifolds, actions)
		return sum(robot_rewards.values()) + sum(task_rewards.values())

	@partial(jax.jit, static_argnames=("self",))
	def get_task_rewards(self, env_state, manifolds, actions):
		raise NotImplementedError

	@partial(jax.jit, static_argnames=("self",))
	def get_terminated(self, env_state, manifolds, actions):
		raise NotImplementedError

	@partial(jax.jit, static_argnames=("self",))
	def get_success(self, env_state, manifolds, actions):
		raise NotImplementedError

	def close(self):
		pass

	@partial(jax.jit, static_argnames=("self",))
	def get_robot_rewards(self, env_state, manifolds, actions):
		action = (
			actions[
				self.static_sim_params.num_joints : self.static_sim_params.num_joints
				+ self.action_space.shape[0]
			]
			/ self.action_max
		)
		return {"energy_penalty": -jnp.sum(action**2)}

	@partial(jax.jit, static_argnames=("self",))
	def get_observation(self, env_state, manifolds, action):
		return {
			"vector": jnp.concatenate(
				[
					2 * env_state.sim_state.polygon.position[self.robot_idx] / self.scene_size - 1.0,
					env_state.sim_state.polygon.velocity[self.robot_idx] / self.sim_params.clip_velocity,
					jnp.cos(jnp.expand_dims(env_state.sim_state.polygon.rotation[self.robot_idx], axis=0)),
					jnp.sin(jnp.expand_dims(env_state.sim_state.polygon.rotation[self.robot_idx], axis=0)),
					jnp.expand_dims(
						env_state.sim_state.polygon.angular_velocity[self.robot_idx], axis=0
					) / self.sim_params.clip_angular_velocity,
				]
			),
			"image": 2 * self.renderer(env_state) - 1.0,
		}

	@partial(jax.jit, static_argnames=("self",))
	def apply_action(self, actions, action):
		action = jnp.clip(action, -1.0, 1.0)
		return actions.at[
			self.static_sim_params.num_joints : self.static_sim_params.num_joints + 4
		].set(self.action_max * action)

	@partial(jax.jit, static_argnames=("self",))
	def get_polygon_position(self, env_state, idx):
		return env_state.sim_state.polygon.position[idx]

	@partial(jax.jit, static_argnames=("self",))
	def get_circle_position(self, env_state, idx):
		return env_state.sim_state.circle.position[idx]

	@partial(jax.jit, static_argnames=("self",))
	def get_polygon_velocity(self, env_state, idx):
		return env_state.sim_state.polygon.velocity[idx]

	@partial(jax.jit, static_argnames=("self",))
	def get_circle_velocity(self, env_state, idx):
		return env_state.sim_state.circle.velocity[idx]

	@partial(jax.jit, static_argnames=("self",))
	def set_polygon_position(self, env_state, idx, position):
		polygon = env_state.sim_state.polygon.replace(
			position=env_state.sim_state.polygon.position.at[idx].set(position)
		)
		return env_state.replace(sim_state=env_state.sim_state.replace(polygon=polygon))

	@partial(jax.jit, static_argnames=("self",))
	def set_circle_position(self, env_state, idx, position):
		circle = env_state.sim_state.circle.replace(
			position=env_state.sim_state.circle.position.at[idx].set(position)
		)
		return env_state.replace(sim_state=env_state.sim_state.replace(circle=circle))

	@partial(jax.jit, static_argnames=("self",))
	def set_polygon_velocity(self, env_state, idx, velocity):
		polygon = env_state.sim_state.polygon.replace(
			velocity=env_state.sim_state.polygon.velocity.at[idx].set(velocity)
		)
		return env_state.replace(sim_state=env_state.sim_state.replace(polygon=polygon))

	@partial(jax.jit, static_argnames=("self",))
	def set_circle_velocity(self, env_state, idx, velocity):
		circle = env_state.sim_state.circle.replace(
			velocity=env_state.sim_state.circle.velocity.at[idx].set(velocity)
		)
		return env_state.replace(sim_state=env_state.sim_state.replace(circle=circle))

	@partial(jax.jit, static_argnames=("self",))
	def set_polygon_rotation(self, env_state, idx, rotation):
		polygon = env_state.sim_state.polygon.replace(
			rotation=env_state.sim_state.polygon.rotation.at[idx].set(rotation)
		)
		return env_state.replace(sim_state=env_state.sim_state.replace(polygon=polygon))

	@partial(jax.jit, static_argnames=("self",))
	def set_circle_rotation(self, env_state, idx, rotation):
		circle = env_state.sim_state.circle.replace(
			rotation=env_state.sim_state.circle.rotation.at[idx].set(rotation)
		)
		return env_state.replace(sim_state=env_state.sim_state.replace(circle=circle))

	@partial(jax.jit, static_argnames=("self",))
	def set_polygon_angular_velocity(self, env_state, idx, angular_velocity):
		polygon = env_state.sim_state.polygon.replace(
			angular_velocity=env_state.sim_state.polygon.angular_velocity.at[idx].set(
				angular_velocity
			)
		)
		return env_state.replace(sim_state=env_state.sim_state.replace(polygon=polygon))

	@partial(jax.jit, static_argnames=("self",))
	def set_circle_angular_velocity(self, env_state, idx, angular_velocity):
		circle = env_state.sim_state.circle.replace(
			angular_velocity=env_state.sim_state.circle.angular_velocity.at[idx].set(
				angular_velocity
			)
		)
		return env_state.replace(sim_state=env_state.sim_state.replace(circle=circle))

	@partial(jax.jit, static_argnames=("self",))
	def remove_polygon(self, env_state, idx):
		position = 1e6 * self.scene_size * jnp.ones(2)
		env_state = self.set_polygon_position(env_state, idx, position)
		return env_state

	@partial(jax.jit, static_argnames=("self",))
	def remove_circle(self, env_state, idx):
		position = 1e6 * self.scene_size * jnp.ones(2)
		env_state = self.set_circle_position(env_state, idx, position)
		return env_state

	@partial(jax.jit, static_argnames=("self",))
	def dist_pp(self, env_state, idxa, idxb):
		return jnp.linalg.norm(
			env_state.sim_state.polygon.position[idxa] - env_state.sim_state.polygon.position[idxb]
		)

	@partial(jax.jit, static_argnames=("self",))
	def dist_cc(self, env_state, idxa, idxb):
		idxa = idxa - self.static_sim_params.num_polygons
		idxb = idxb - self.static_sim_params.num_polygons
		return jnp.linalg.norm(
			env_state.sim_state.circle.position[idxa] - env_state.sim_state.circle.position[idxb]
		)

	@partial(jax.jit, static_argnames=("self",))
	def dist_cp(self, env_state, idxa, idxb):
		idxa = idxa - self.static_sim_params.num_polygons
		return jnp.linalg.norm(
			env_state.sim_state.circle.position[idxa] - env_state.sim_state.polygon.position[idxb]
		)

	@partial(jax.jit, static_argnames=("self",))
	def collision_pp(self, manifolds, idxa, idxb):
		def get_active(manifold):
			return manifold.active

		pair = (
			(self.physics_engine.poly_poly_pairs[:, 0] == idxa)
			& (self.physics_engine.poly_poly_pairs[:, 1] == idxb)
		) | (
			(self.physics_engine.poly_poly_pairs[:, 0] == idxb)
			& (self.physics_engine.poly_poly_pairs[:, 1] == idxa)
		)
		return jnp.any(pair * get_active(manifolds[0]))

	@partial(jax.jit, static_argnames=("self",))
	def collision_cc(self, manifolds, idxa, idxb):
		def get_active(manifold):
			return manifold.active

		idxa = idxa - self.static_sim_params.num_polygons
		idxb = idxb - self.static_sim_params.num_polygons

		pair = (
			(self.physics_engine.circle_circle_pairs[:, 0] == idxa)
			& (self.physics_engine.circle_circle_pairs[:, 1] == idxb)
		) | (
			(self.physics_engine.circle_circle_pairs[:, 0] == idxb)
			& (self.physics_engine.circle_circle_pairs[:, 1] == idxa)
		)
		return jnp.any(pair * get_active(manifolds[2]))

	@partial(jax.jit, static_argnames=("self",))
	def collision_cp(self, manifolds, idxa, idxb):
		def get_active(manifold):
			return manifold.active

		pair = (
			(self.physics_engine.circle_poly_pairs[:, 0] == (idxa - self.static_sim_params.num_polygons))
			& (self.physics_engine.circle_poly_pairs[:, 1] == idxb)
		) | (
			(self.physics_engine.circle_poly_pairs[:, 0] == (idxb - self.static_sim_params.num_polygons))
			& (self.physics_engine.circle_poly_pairs[:, 1] == idxa)
		)
		return jnp.any(pair * get_active(manifolds[1]))

	@partial(jax.jit, static_argnames=("self",))
	def get_manifolds(self, env_state):
		return (
			jax.tree.map(lambda x: x[:, 0], env_state.sim_state.acc_rr_manifolds),
			env_state.sim_state.acc_cr_manifolds,
			env_state.sim_state.acc_cc_manifolds,
		)

	@partial(jax.jit, static_argnames=("self",))
	def add_rectangle_to_scene(
		self,
		env_state,
		position,
		dimensions,
		rotation=0.0,
		velocity=jnp.zeros(2),
		angular_velocity=0.0,
		density=1.0,
		friction=1.0,
		restitution=0.0,
		fixated=False,
		color=jnp.ones(3),
	):
		sim_state, (_, idx) = add_rectangle_to_scene(
			sim_state=env_state.sim_state,
			static_sim_params=self.static_sim_params,
			position=position,
			dimensions=dimensions,
			rotation=rotation,
			velocity=velocity,
			angular_velocity=angular_velocity,
			density=density,
			friction=friction,
			restitution=restitution,
			fixated=fixated,
			color=color,
		)
		env_state = env_state.replace(sim_state=sim_state)
		return env_state, idx

	# @partial(jax.jit, static_argnames=("self",))
	def add_circle_to_scene(
		self,
		env_state,
		position,
		radius,
		rotation=0.0,
		velocity=jnp.zeros(2),
		angular_velocity=0.0,
		density=1.0,
		friction=1.0,
		restitution=0.0,
		fixated=False,
		color=jnp.ones(3),
	):
		sim_state, (_, idx) = add_circle_to_scene(
			sim_state=env_state.sim_state,
			static_sim_params=self.static_sim_params,
			position=position,
			radius=radius,
			rotation=rotation,
			velocity=velocity,
			angular_velocity=angular_velocity,
			density=density,
			friction=friction,
			restitution=restitution,
			fixated=fixated,
			color=color,
		)
		env_state = env_state.replace(sim_state=sim_state)
		return env_state, idx

	@partial(jax.jit, static_argnames=("self",))
	def add_fixed_joint_to_scene(
		self,
		env_state,
		a_index,
		b_index,
		a_relative_pos,
		b_relative_pos,
	):
		sim_state, idx = add_fixed_joint_to_scene(
			sim_state=env_state.sim_state,
			static_sim_params=self.static_sim_params,
			a_index=a_index,
			b_index=b_index,
			a_relative_pos=a_relative_pos,
			b_relative_pos=b_relative_pos,
		)
		env_state = env_state.replace(sim_state=sim_state)
		return env_state, idx

	@partial(jax.jit, static_argnames=("self",))
	def add_thruster_to_scene(
		self,
		env_state,
		object_index,
		relative_position,
		rotation,
		power=1.0,
	):
		sim_state, idx = add_thruster_to_scene(
			sim_state=env_state.sim_state,
			object_index=object_index,
			relative_position=relative_position,
			rotation=rotation,
			power=power,
		)
		env_state = env_state.replace(sim_state=sim_state)
		return env_state, idx

	def visualize(self):
		key = jax.random.PRNGKey(0)
		env_state = self.reset(key)
		renders = []
		for _ in range(100):
			env_state = self.step(key, env_state, 0.0 * self.action_space.sample())
			renders.append(self.renderer(env_state))
		return renders


def make_render(static_sim_params, screen_dim, scene_size):
	ppud = screen_dim[0] // scene_size  # pixels per unit distance
	patch_size = 512
	screen_padding = patch_size
	full_screen_size = (
		screen_dim[0] + 2 * screen_padding,
		screen_dim[1] + 2 * screen_padding,
	)
	background_color = jnp.array([0.9, 0.9, 0.9])

	def _world_space_to_pixel_space(x):
		return x * ppud + screen_padding

	cleared_screen = clear_screen(full_screen_size, background_color)

	circle_shader = add_mask_to_shader(fragment_shader_circle)
	circle_renderer = make_renderer(
		full_screen_size, circle_shader, (patch_size, patch_size), batched=True
	)

	polygon_shader = add_mask_to_shader(
		make_fragment_shader_convex_dynamic_ngon_with_edges(static_sim_params.max_polygon_vertices)
	)
	polygon_renderer = make_renderer(
		full_screen_size, polygon_shader, (patch_size, patch_size), batched=True
	)

	@jax.jit
	def render(env_state):
		sim_state = env_state.sim_state
		pixels = cleared_screen

		# Circles
		circle_positions_pixel_space = _world_space_to_pixel_space(sim_state.circle.position)
		circle_radii_pixel_space = sim_state.circle.radius * ppud
		circle_patch_positions = (circle_positions_pixel_space - (patch_size / 2)).astype(jnp.int32)
		circle_patch_positions = jnp.maximum(circle_patch_positions, 0)

		circle_uniforms = (
			circle_positions_pixel_space,
			circle_radii_pixel_space,
			sim_state.circle.color,
			sim_state.circle.active,
		)

		pixels = circle_renderer(pixels, circle_patch_positions, circle_uniforms)

		# Rectangles
		polygon_positions_pixel_space = _world_space_to_pixel_space(sim_state.polygon.position)
		polygon_rmats = jax.vmap(rmat)(sim_state.polygon.rotation)
		polygon_rmats = jnp.repeat(
			polygon_rmats[:, None, :, :],
			repeats=static_sim_params.max_polygon_vertices,
			axis=1,
		)
		polygon_vertices_pixel_space = _world_space_to_pixel_space(
			sim_state.polygon.position[:, None, :]
			+ jax.vmap(jax.vmap(jnp.matmul))(polygon_rmats, sim_state.polygon.vertices)
		)
		polygon_patch_positions = (polygon_positions_pixel_space - (patch_size / 2)).astype(
			jnp.int32
		)
		polygon_patch_positions = jnp.maximum(polygon_patch_positions, 0)

		polygon_uniforms = (
			polygon_vertices_pixel_space,
			sim_state.polygon.color,
			sim_state.polygon.color,
			sim_state.polygon.n_vertices,
			sim_state.polygon.active,
		)

		pixels = polygon_renderer(pixels, polygon_patch_positions, polygon_uniforms)

		# Crop out the sides
		cropped = pixels[screen_padding:-screen_padding, screen_padding:-screen_padding]

		return jnp.rot90(cropped, k=1)

	return render
