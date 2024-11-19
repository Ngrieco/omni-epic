import jax.numpy as jnp
from jax2d.engine import (
	calculate_collision_matrix,
	get_empty_collision_manifolds,
	recalculate_mass_and_inertia,
)
from jax2d.sim_state import Joint, Thruster, SimState

from omni_epic.jax2d.sim_state import RigidBody


def create_empty_sim(
	static_sim_params,
	add_floor=True,
	add_walls_and_ceiling=True,
	scene_size=5,
	floor_offset=0.2,
):
	# Polygons
	polygon_pos = jnp.zeros((static_sim_params.num_polygons, 2), dtype=jnp.float32)
	polygon_vertices = jnp.zeros(
		(static_sim_params.num_polygons, static_sim_params.max_polygon_vertices, 2),
		dtype=jnp.float32,
	)
	polygon_vel = jnp.zeros((static_sim_params.num_polygons, 2), dtype=jnp.float32)
	polygon_rotation = jnp.zeros(static_sim_params.num_polygons, dtype=jnp.float32)
	polygon_angular_velocity = jnp.zeros(static_sim_params.num_polygons, dtype=jnp.float32)
	polygon_inverse_mass = jnp.zeros(static_sim_params.num_polygons, dtype=jnp.float32)
	polygon_inverse_inertia = jnp.zeros(static_sim_params.num_polygons, dtype=jnp.float32)
	polygon_active = jnp.zeros(static_sim_params.num_polygons, dtype=bool)

	# Circles
	circle_position = jnp.zeros((static_sim_params.num_circles, 2), dtype=jnp.float32)
	circle_radius = jnp.zeros(static_sim_params.num_circles, dtype=jnp.float32)
	circle_vel = jnp.zeros((static_sim_params.num_circles, 2), dtype=jnp.float32)
	circle_inverse_mass = jnp.zeros(static_sim_params.num_circles, dtype=jnp.float32)
	circle_inverse_inertia = jnp.zeros(static_sim_params.num_circles, dtype=jnp.float32)
	circle_active = jnp.zeros(static_sim_params.num_circles, dtype=bool)

	# We simulate half-spaces by just using polygons with large dimensions.
	# Floor
	if add_floor:
		polygon_pos = polygon_pos.at[0].set(jnp.array([scene_size / 2, -scene_size + floor_offset]))
		polygon_vertices = polygon_vertices.at[0].set(
			jnp.array(
				[
					[scene_size / 2, scene_size + floor_offset],
					[scene_size / 2, -scene_size - floor_offset],
					[-scene_size / 2, -scene_size - floor_offset],
					[-scene_size / 2, scene_size + floor_offset],
				]
			)
		)

		polygon_inverse_mass = polygon_inverse_mass.at[0].set(0.0)
		polygon_inverse_inertia = polygon_inverse_inertia.at[0].set(0.0)
		polygon_active = polygon_active.at[0].set(True)

	if add_walls_and_ceiling:
		# Side Walls
		polygon_pos = polygon_pos.at[1].set(jnp.array([0.0, 0.0]))
		polygon_vertices = polygon_vertices.at[1].set(
			jnp.array(
				[
					[-scene_size, scene_size],
					[-0.05, scene_size],
					[-0.05, 0],
					[-scene_size, 0],
				]
			)
		)

		polygon_inverse_mass = polygon_inverse_mass.at[1].set(0.0)
		polygon_inverse_inertia = polygon_inverse_inertia.at[1].set(0.0)

		polygon_pos = polygon_pos.at[2].set(jnp.array([0.0, 0.0]))
		polygon_vertices = polygon_vertices.at[2].set(
			jnp.array(
				[
					[scene_size, scene_size],
					[2 * scene_size, scene_size],
					[2 * scene_size, 0],
					[scene_size, 0],
				]
			)
		)

		polygon_inverse_mass = polygon_inverse_mass.at[2].set(0.0)
		polygon_inverse_inertia = polygon_inverse_inertia.at[2].set(0.0)

		# Ceiling
		polygon_pos = polygon_pos.at[3].set(
			jnp.array([scene_size / 2, floor_offset + 2 * scene_size])
		)
		polygon_vertices = polygon_vertices.at[3].set(
			jnp.array(
				[
					[scene_size, scene_size + floor_offset],
					[scene_size / 2, -scene_size - floor_offset],
					[-(scene_size / 2), -scene_size - floor_offset],
					[-(scene_size / 2), scene_size + floor_offset],
				]
			)
		)

		polygon_inverse_mass = polygon_inverse_mass.at[3].set(0.0)
		polygon_inverse_inertia = polygon_inverse_inertia.at[3].set(0.0)

		polygon_active = polygon_active.at[1:4].set(True)

	# Joints
	revolute_joint_a_pos = jnp.zeros((static_sim_params.num_joints, 2), dtype=jnp.float32)
	revolute_joint_b_pos = jnp.zeros((static_sim_params.num_joints, 2), dtype=jnp.float32)
	revolute_joint_a_index = jnp.zeros(static_sim_params.num_joints, dtype=jnp.int32)
	revolute_joint_b_index = jnp.zeros(static_sim_params.num_joints, dtype=jnp.int32)

	joints = Joint(
		a_index=revolute_joint_a_index,
		b_index=revolute_joint_b_index,
		a_relative_pos=revolute_joint_a_pos,
		b_relative_pos=revolute_joint_b_pos,
		active=jnp.zeros(static_sim_params.num_joints, dtype=bool),
		global_position=jnp.ones((static_sim_params.num_joints, 2), dtype=jnp.float32) * 256,
		motor_on=jnp.zeros(static_sim_params.num_joints, dtype=bool),
		motor_speed=jnp.zeros(static_sim_params.num_joints, dtype=jnp.float32),
		motor_power=jnp.zeros(static_sim_params.num_joints, dtype=jnp.float32),
		acc_impulse=jnp.zeros((static_sim_params.num_joints, 2), dtype=jnp.float32),
		motor_has_joint_limits=jnp.zeros(static_sim_params.num_joints, dtype=bool),
		min_rotation=jnp.zeros(static_sim_params.num_joints, dtype=jnp.float32),
		max_rotation=jnp.zeros(static_sim_params.num_joints, dtype=jnp.float32),
		is_fixed_joint=jnp.zeros(static_sim_params.num_joints, dtype=bool),
		rotation=jnp.zeros(static_sim_params.num_joints, dtype=jnp.float32),
		acc_r_impulse=jnp.zeros((static_sim_params.num_joints), dtype=jnp.float32),
	)

	thrusters = Thruster(
		object_index=jnp.zeros(static_sim_params.num_thrusters, dtype=jnp.int32),
		relative_position=jnp.zeros((static_sim_params.num_thrusters, 2), dtype=jnp.float32),
		active=jnp.zeros(static_sim_params.num_thrusters, dtype=bool),
		power=jnp.zeros(static_sim_params.num_thrusters, dtype=jnp.float32),
		global_position=jnp.zeros((static_sim_params.num_thrusters, 2), dtype=jnp.float32),
		rotation=jnp.zeros(static_sim_params.num_thrusters, dtype=jnp.float32),
	)

	collision_matrix = calculate_collision_matrix(static_sim_params, joints)

	(
		acc_rr_manifolds,
		acc_cr_manifolds,
		acc_cc_manifolds,
	) = get_empty_collision_manifolds(static_sim_params)

	n_vertices = jnp.ones((static_sim_params.num_polygons,), dtype=jnp.int32) * 4
	state = SimState(
		polygon=RigidBody(
			position=polygon_pos,
			vertices=polygon_vertices,
			n_vertices=n_vertices,
			velocity=polygon_vel * 0,
			inverse_mass=polygon_inverse_mass,
			rotation=polygon_rotation,
			angular_velocity=polygon_angular_velocity,
			inverse_inertia=polygon_inverse_inertia,
			friction=jnp.ones(static_sim_params.num_polygons),
			restitution=jnp.zeros(static_sim_params.num_polygons, dtype=jnp.float32),
			radius=jnp.zeros(static_sim_params.num_polygons, dtype=jnp.float32),
			active=polygon_active,
			collision_mode=jnp.ones(static_sim_params.num_polygons, dtype=int)
			.at[: static_sim_params.num_static_fixated_polys]
			.set(2),
			color=jnp.zeros((static_sim_params.num_polygons, 3), dtype=jnp.float32),
		),
		circle=RigidBody(
			radius=circle_radius,
			position=circle_position,
			velocity=circle_vel,
			inverse_mass=circle_inverse_mass,
			inverse_inertia=circle_inverse_inertia,
			rotation=jnp.ones(static_sim_params.num_circles) * 0,
			angular_velocity=jnp.ones(static_sim_params.num_circles) * 0,
			friction=jnp.ones(static_sim_params.num_circles),
			restitution=jnp.zeros(static_sim_params.num_circles, dtype=jnp.float32),
			vertices=jnp.zeros(
				(
					static_sim_params.num_circles,
					static_sim_params.max_polygon_vertices,
					2,
				),
				dtype=jnp.float32,
			),
			n_vertices=jnp.zeros((static_sim_params.num_circles,), dtype=jnp.int32),
			active=circle_active,
			collision_mode=jnp.ones(static_sim_params.num_circles, dtype=int),
			color=jnp.zeros((static_sim_params.num_polygons, 3), dtype=jnp.float32),
		),
		thruster=thrusters,
		joint=joints,
		collision_matrix=collision_matrix,
		acc_rr_manifolds=acc_rr_manifolds,
		acc_cr_manifolds=acc_cr_manifolds,
		acc_cc_manifolds=acc_cc_manifolds,
		gravity=jnp.array([0.0, -9.81]),
	)

	polygon_densities = jnp.ones(static_sim_params.num_polygons)
	circle_densities = jnp.ones(static_sim_params.num_circles)

	state = recalculate_mass_and_inertia(
		state, static_sim_params, polygon_densities, circle_densities
	)

	return state
