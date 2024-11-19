import importlib.util
from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from omni_epic.envs.base import EnvState, EnvBase as Env


class Wrapper:
	"""Wraps an environment to allow modular transformations."""

	def __init__(self, env: Env):
		self.env = env

	def reset(self, key: jax.Array) -> EnvState:
		return self.env.reset(key)

	def step(self, key: jax.Array, state: EnvState, action: jax.Array) -> EnvState:
		return self.env.step(key, state, action)

	@property
	def observation_size(self) -> int:
		return self.env.observation_size

	@property
	def action_size(self) -> int:
		return self.env.action_size

	@property
	def unwrapped(self) -> Env:
		return self.env.unwrapped

	@property
	def backend(self) -> str:
		return self.unwrapped.backend

	def __getattr__(self, name):
		if name == "__setstate__":
			raise AttributeError(name)
		return getattr(self.env, name)


@struct.dataclass
class LogEnvState:
	env_state: EnvState
	episode_returns: float
	episode_lengths: int
	returned_episode_returns: float
	returned_episode_lengths: int
	timestep: int


class LogWrapper(Wrapper):
	"""Log the episode returns and lengths."""

	def __init__(self, env: Env):
		super().__init__(env)

	@partial(jax.jit, static_argnums=(0,))
	def reset(self, key: jax.Array) -> LogEnvState:
		env_state = self.env.reset(key)
		env_state = LogEnvState(env_state, 0, 0, 0, 0, 0)
		return env_state

	@partial(jax.jit, static_argnums=(0,))
	def step(
		self, key: jax.Array, env_state: EnvState, action: jax.Array
	) -> tuple[chex.Array, environment.EnvState, float, bool, dict]:
		env_state, reward, terminated = self.env.step(key, env_state, action)
		new_episode_return = state.episode_returns + reward
		new_episode_length = state.episode_lengths + 1
		state = LogEnvState(
			env_state=env_state,
			episode_returns=new_episode_return * (1 - terminated),
			episode_lengths=new_episode_length * (1 - terminated),
			returned_episode_returns=state.returned_episode_returns * (1 - terminated)
			+ new_episode_return * terminated,
			returned_episode_lengths=state.returned_episode_lengths * (1 - terminated)
			+ new_episode_length * terminated,
			timestep=state.timestep + 1,
		)
		info["returned_episode_returns"] = state.returned_episode_returns
		info["returned_episode_lengths"] = state.returned_episode_lengths
		info["timestep"] = state.timestep
		info["returned_episode"] = terminated
		return env_state, reward, terminated, info


class Jax2DWrapper(Wrapper):
	"""Wraps a Jax2D environment."""

	def __init__(self, env_path):
		# Load the environment
		spec = importlib.util.spec_from_file_location("env", env_path)
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
		Env = getattr(module, "Env")
		self.env = Env()


class EpisodeWrapper(Wrapper):
	"""Maintains episode step count and sets done at episode end."""

	def __init__(self, env: Env, episode_length: int, action_repeat: int):
		super().__init__(env)
		self.episode_length = episode_length
		self.action_repeat = action_repeat

	def reset(self, key: jax.Array) -> EnvState:
		env_state = self.env.reset(key)
		env_state.info["steps"] = jnp.array(0.0)
		env_state.info["truncation"] = jnp.array(0.0)
		return env_state

	def step(self, key: jax.Array, env_state: EnvState, action: jax.Array) -> EnvState:
		def f(env_state, _):
			env_state = self.env.step(key, env_state, action)
			return env_state, (env_state.reward, env_state.terminated)

		env_state, (reward_, terminated_) = jax.lax.scan(f, env_state, (), self.action_repeat)

		one = jnp.ones_like(terminated_[-1])
		zero = jnp.zeros_like(terminated_[-1])
		steps = env_state.info["steps"] + self.action_repeat
		episode_length = jnp.array(self.episode_length, dtype=jnp.int32)

		terminated_any = jnp.clip(jnp.sum(terminated_, axis=0), 0.0, 1.0)
		terminated = jnp.where(steps >= episode_length, one, terminated_any)
		env_state.info["truncation"] = jnp.where(steps >= episode_length, 1 - terminated_any, zero)

		env_state.info["steps"] = steps

		return env_state.replace(reward=jnp.sum(reward_, axis=0), terminated=terminated)


class AutoResetWrapper(Wrapper):
	"""Automatically resets envs that are done."""

	def reset(self, key: jax.Array) -> EnvState:
		env_state = self.env.reset(key)
		env_state.info['sim_state'] = env_state.sim_state
		return env_state

	def step(self, key: jax.Array, env_state: EnvState, action: jax.Array) -> EnvState:
		if "steps" in env_state.info:
			steps = env_state.info["steps"]
			steps = jnp.where(env_state.terminated, jnp.zeros_like(steps), steps)
			env_state.info.update(steps=steps)

		env_state = env_state.replace(terminated=jnp.zeros_like(env_state.terminated))
		env_state = self.env.step(key, env_state, action)

		def where_done(x, y):
			terminated = env_state.terminated
			if terminated.shape:
				terminated = jnp.reshape(terminated, [x.shape[0]] + [1] * (len(x.shape) - 1))
			return jnp.where(terminated, x, y)

		sim_state = jax.tree.map(
			where_done, env_state.info["sim_state"], env_state.sim_state
		)
		return env_state.replace(sim_state=sim_state)


class GymnaxWrapper:
	pass


class ClipAction(GymnaxWrapper):
	def __init__(self, env, low=-1.0, high=1.0):
		super().__init__(env)
		self.low = low
		self.high = high

	def step(self, key, state, action, params=None):
		"""TODO: In theory the below line should be the way to do this."""
		# action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
		action = jnp.clip(action, self.low, self.high)
		return self._env.step(key, state, action, params)


class TransformObservation(GymnaxWrapper):
	def __init__(self, env, transform_obs):
		super().__init__(env)
		self.transform_obs = transform_obs

	def reset(self, key, params=None):
		obs, state = self._env.reset(key, params)
		return self.transform_obs(obs), state

	def step(self, key, state, action, params=None):
		obs, state, reward, done, info = self._env.step(key, state, action, params)
		return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
	def __init__(self, env, transform_reward):
		super().__init__(env)
		self.transform_reward = transform_reward

	def step(self, key, state, action, params=None):
		obs, state, reward, done, info = self._env.step(key, state, action, params)
		return obs, state, self.transform_reward(reward), done, info


class VecEnv(GymnaxWrapper):
	def __init__(self, env):
		super().__init__(env)
		self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
		self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
	mean: jnp.ndarray
	var: jnp.ndarray
	count: float
	env_state: environment.EnvState


class NormalizeVecObservation(GymnaxWrapper):
	def __init__(self, env):
		super().__init__(env)

	def reset(self, key, params=None):
		obs, state = self._env.reset(key, params)
		state = NormalizeVecObsEnvState(
			mean=jnp.zeros_like(obs),
			var=jnp.ones_like(obs),
			count=1e-4,
			env_state=state,
		)
		batch_mean = jnp.mean(obs, axis=0)
		batch_var = jnp.var(obs, axis=0)
		batch_count = obs.shape[0]

		delta = batch_mean - state.mean
		tot_count = state.count + batch_count

		new_mean = state.mean + delta * batch_count / tot_count
		m_a = state.var * state.count
		m_b = batch_var * batch_count
		M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
		new_var = M2 / tot_count
		new_count = tot_count

		state = NormalizeVecObsEnvState(
			mean=new_mean,
			var=new_var,
			count=new_count,
			env_state=state.env_state,
		)

		return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

	def step(self, key, state, action, params=None):
		obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)

		batch_mean = jnp.mean(obs, axis=0)
		batch_var = jnp.var(obs, axis=0)
		batch_count = obs.shape[0]

		delta = batch_mean - state.mean
		tot_count = state.count + batch_count

		new_mean = state.mean + delta * batch_count / tot_count
		m_a = state.var * state.count
		m_b = batch_var * batch_count
		M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
		new_var = M2 / tot_count
		new_count = tot_count

		state = NormalizeVecObsEnvState(
			mean=new_mean,
			var=new_var,
			count=new_count,
			env_state=env_state,
		)
		return (
			(obs - state.mean) / jnp.sqrt(state.var + 1e-8),
			state,
			reward,
			done,
			info,
		)


@struct.dataclass
class NormalizeVecRewEnvState:
	mean: jnp.ndarray
	var: jnp.ndarray
	count: float
	return_val: float
	env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
	def __init__(self, env, gamma):
		super().__init__(env)
		self.gamma = gamma

	def reset(self, key, params=None):
		obs, state = self._env.reset(key, params)
		batch_count = obs.shape[0]
		state = NormalizeVecRewEnvState(
			mean=0.0,
			var=1.0,
			count=1e-4,
			return_val=jnp.zeros((batch_count,)),
			env_state=state,
		)
		return obs, state

	def step(self, key, state, action, params=None):
		obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
		return_val = state.return_val * self.gamma * (1 - done) + reward

		batch_mean = jnp.mean(return_val, axis=0)
		batch_var = jnp.var(return_val, axis=0)
		batch_count = obs.shape[0]

		delta = batch_mean - state.mean
		tot_count = state.count + batch_count

		new_mean = state.mean + delta * batch_count / tot_count
		m_a = state.var * state.count
		m_b = batch_var * batch_count
		M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
		new_var = M2 / tot_count
		new_count = tot_count

		state = NormalizeVecRewEnvState(
			mean=new_mean,
			var=new_var,
			count=new_count,
			return_val=return_val,
			env_state=env_state,
		)
		return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info
