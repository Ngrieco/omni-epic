import importlib.util

import jax
import jax.numpy as jnp

from omni_epic.envs.base import EnvState, EnvBase as Env


class Wrapper:
	"""Wraps an environment to allow modular transformations."""

	def __init__(self, env: Env):
		self.env = env

	def reset(self, key: jax.Array) -> EnvState:
		return self.env.reset(key)

	def step(self, key: jax.Array, state: EnvState, action: jax.Array) -> EnvState:
		return self.env.step(key, state, action)

	def __getattr__(self, name):
		if name == "__setstate__":
			raise AttributeError(name)
		return getattr(self.env, name)


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
		env_state.info["first_sim_state"] = env_state.sim_state
		env_state.info["first_observation"] = env_state.observation
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
			where_done, env_state.info["first_sim_state"], env_state.sim_state
		)
		observation = jax.tree.map(
			where_done, env_state.info["first_observation"], env_state.observation
		)
		return env_state.replace(sim_state=sim_state, observation=observation)


class LogWrapper(Wrapper):
	"""Log the episode returns and lengths."""

	def __init__(self, env: Env):
		super().__init__(env)

	def reset(self, key: jax.Array) -> EnvState:
		env_state = self.env.reset(key)
		env_state.info.update(
			episode_returns=jnp.zeros_like(env_state.reward),
			episode_lengths=jnp.zeros_like(env_state.reward, dtype=jnp.int32),
			returned_episode_returns=jnp.zeros_like(env_state.reward),
			returned_episode_lengths=jnp.zeros_like(env_state.reward, dtype=jnp.int32),
			timestep=jnp.zeros_like(env_state.reward, dtype=jnp.int32),
		)
		return env_state

	def step(self, key: jax.Array, env_state: EnvState, action: jax.Array) -> EnvState:
		env_state = self.env.step(key, env_state, action)
		new_episode_return = env_state.info["episode_returns"] + env_state.reward
		new_episode_length = env_state.info["episode_lengths"] + 1

		env_state.info.update(
			episode_returns=new_episode_return * (1 - env_state.terminated),
			episode_lengths=new_episode_length * (1 - env_state.terminated),
			returned_episode_returns=env_state.info["returned_episode_returns"] * (1 - env_state.terminated)
			+ new_episode_return * env_state.terminated,
			returned_episode_lengths=env_state.info["returned_episode_lengths"] * (1 - env_state.terminated)
			+ new_episode_length * env_state.terminated,
			timestep=env_state.info["timestep"] + 1,
		)
		return env_state


class ClipAction(Wrapper):
	def __init__(self, env, low=-1.0, high=1.0):
		super().__init__(env)
		self.low = low
		self.high = high

	def step(self, key, env_state, action):
		action = jnp.clip(action, self.low, self.high)
		return self.env.step(key, env_state, action)


class TransformObservation(Wrapper):
	def __init__(self, env, transform_obs):
		super().__init__(env)
		self.transform_observation = transform_obs

	def reset(self, key):
		env_state = self.env.reset(key)
		return env_state.replace(observation=self.transform_observation(env_state.observation))

	def step(self, key, env_state, action):
		env_state = self.env.step(key, env_state, action)
		return env_state.replace(observation=self.transform_observation(env_state.observation))


class TransformReward(Wrapper):
	def __init__(self, env, transform_reward):
		super().__init__(env)
		self.transform_reward = transform_reward

	def step(self, key, env_state, action):
		env_state = self.env.step(key, env_state, action)
		return env_state.replace(reward=self.transform_reward(env_state.reward))


class VecEnv(Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.reset = jax.vmap(self.env.reset)
		self.step = jax.vmap(self.env.step)


class NormalizeVecObservation(Wrapper):
	def __init__(self, env):
		super().__init__(env)

	def reset(self, key):
		env_state = self.env.reset(key)

		# Initialize observation statistics
		env_state.info["observation_mean"] = jax.tree.map(jnp.zeros_like, env_state.observation)
		env_state.info["observation_var"] = jax.tree.map(jnp.ones_like, env_state.observation)
		env_state.info["observation_count"] = jnp.array(1e-4)

		# Compute batch statistics
		batch_mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), env_state.observation)
		batch_var = jax.tree.map(lambda x: jnp.var(x, axis=0), env_state.observation)
		batch_count = jax.tree.leaves(env_state.observation)[0].shape[0]

		delta = jax.tree.map(lambda x, y: x - y, batch_mean, env_state.info["observation_mean"])
		tot_count = env_state.info["observation_count"] + batch_count

		new_mean = jax.tree.map(lambda x, y: x + y * batch_count / tot_count, env_state.info["observation_mean"], delta)
		m_a = jax.tree.map(lambda x: x * env_state.info["observation_count"], env_state.info["observation_var"])
		m_b = jax.tree.map(lambda x: x * batch_count, batch_var)
		m2 = jax.tree.map(
			lambda x, y, z: x + y + jnp.square(z) * env_state.info["observation_count"] * batch_count / tot_count,
			m_a,
			m_b,
			delta
		)
		new_var = jax.tree.map(lambda x: x / tot_count, m2)
		new_count = tot_count

		# Normalize observation
		normalized_obs = jax.tree.map(
			lambda x, y, z: (x - y) / jnp.sqrt(z + 1e-8),
			env_state.observation,
			new_mean,
			new_var,
		)

		env_state.info.update(
			observation_mean=new_mean,
			observation_var=new_var,
			observation_count=new_count,
		)
		return env_state.replace(observation=normalized_obs)

	def step(self, key, env_state, action):
		env_state = self.env.step(key, env_state, action)

		batch_mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), env_state.observation)
		batch_var = jax.tree.map(lambda x: jnp.var(x, axis=0), env_state.observation)
		batch_count = jax.tree.leaves(env_state.observation)[0].shape[0]

		delta = jax.tree.map(lambda x, y: x - y, batch_mean, env_state.info["observation_mean"])
		tot_count = env_state.info["observation_count"] + batch_count

		new_mean = jax.tree.map(lambda x, y: x + y * batch_count / tot_count, env_state.info["observation_mean"], delta)
		m_a = jax.tree.map(lambda x, y: x * y, env_state.info["observation_var"], env_state.info["observation_count"])
		m_b = jax.tree.map(lambda x, y: x * y, batch_var, batch_count)
		m2 = jax.tree.map(
			lambda x, y, z: x + y + jnp.square(z) * env_state.info["observation_count"] * batch_count / tot_count,
			m_a,
			m_b,
			delta
		)
		new_var = jax.tree.map(lambda x: x / tot_count, m2)
		new_count = tot_count

		# Normalize observation
		normalized_obs = jax.tree.map(
			lambda x, y, z: (x - y) / jnp.sqrt(z + 1e-8),
			env_state.observation,
			new_mean,
			new_var,
		)

		env_state.info.update(
			observation_mean=new_mean,
			observation_var=new_var,
			observation_count=new_count,
		)
		return env_state.replace(observation=normalized_obs)


class NormalizeVecReward(Wrapper):
	def __init__(self, env, gamma):
		super().__init__(env)
		self.gamma = gamma

	def reset(self, key):
		env_state = self.env.reset(key)
		batch_count = jax.tree.leaves(env_state.observation)[0].shape[0]
		env_state.info.update(
			reward_mean=0.0,
			reward_var=1.0,
			reward_count=1e-4,
			reward_return_val=jnp.zeros((batch_count,)),
		)
		return env_state

	def step(self, key, env_state, action):
		env_state = self.env.step(key, env_state, action)
		return_val = env_state.info["reward_return_val"] * self.gamma * (1 - env_state.terminated) + env_state.reward

		batch_mean = jnp.mean(return_val, axis=0)
		batch_var = jnp.var(return_val, axis=0)
		batch_count = jax.tree.leaves(env_state.observation)[0].shape[0]

		delta = batch_mean - env_state.info["reward_mean"]
		tot_count = env_state.info["reward_count"] + batch_count

		new_mean = env_state.info["reward_mean"] + delta * batch_count / tot_count
		m_a = env_state.info["reward_var"] * env_state.info["reward_count"]
		m_b = batch_var * batch_count
		M2 = m_a + m_b + jnp.square(delta) * env_state.info["reward_count"] * batch_count / tot_count
		new_var = M2 / tot_count
		new_count = tot_count

		normalized_reward = env_state.reward / jnp.sqrt(env_state.info["reward_var"] + 1e-8)

		env_state.info.update(
			reward_mean=new_mean,
			reward_var=new_var,
			reward_count=new_count,
			reward_return_val=return_val,
		)
		return env_state.replace(reward=normalized_reward)
