import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RigidBody:
	position: jnp.ndarray  # Centroid
	rotation: float  # Radians
	velocity: jnp.ndarray  # m/s
	angular_velocity: float  # rad/s

	inverse_mass: (
		float  # We use 0 to denote a fixated object with infinite mass (constant velocity)
	)
	inverse_inertia: (
		float  # Similarly, 0 denotes an object with infinite inertia (constant angular velocity)
	)

	friction: float
	restitution: float  # Due to baumgarte, the actual restitution is a bit higher, so setting this to 1 will cause energy to be created on collision

	collision_mode: int  # 0 == doesn't collide with 1's. 1 = normal, i.e., it collides. 2 == collides with everything (including 0's).
	active: bool

	# Polygon
	n_vertices: int  # >=3 or things blow up
	vertices: jnp.ndarray  # Clockwise or things blow up

	# Circle
	radius: float

	# Color
	color: jnp.ndarray  # RGB
