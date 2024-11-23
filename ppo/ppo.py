from collections.abc import Sequence
from typing import NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from ppo.wrappers import (
	ClipAction,
	Jax2DWrapper,
	EpisodeWrapper,
	AutoResetWrapper,
	LogWrapper,
	NormalizeVecObservation,
	NormalizeVecReward,
	VecEnv,
)


class ActorCritic(nn.Module):
	action_dim: Sequence[int]

	@nn.compact
	def __call__(self, x):  # 64 x 64 x 3
		# Shared CNN
		hidden = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), kernel_init=orthogonal(np.sqrt(2)))(x["image"])  # 16 x 16 x 32
		hidden = nn.relu(hidden)
		hidden = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), kernel_init=orthogonal(np.sqrt(2)))(hidden)  # 8 x 8 x 64
		hidden = nn.relu(hidden)
		hidden = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), kernel_init=orthogonal(np.sqrt(2)))(hidden)  # 8 x 8 x 64
		hidden = nn.relu(hidden)
		hidden = jnp.reshape(hidden, hidden.shape[:-3] + (-1,))  # Flatten 4096 = 8 * 8 * 64
		hidden = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(hidden)
		hidden = nn.relu(hidden)

		# Combine image and vector
		hidden = jnp.concatenate([hidden, x["vector"]], axis=-1)

		# Actor
		actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(hidden)
		actor_mean = nn.relu(actor_mean)
		actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
			actor_mean
		)
		actor_mean = nn.relu(actor_mean)
		actor_mean = nn.Dense(
			self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
		)(actor_mean)
		actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
		pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

		# Critic
		critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(hidden)
		critic = nn.relu(critic)
		critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
		critic = nn.relu(critic)
		critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

		return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
	obs: jnp.ndarray
	action: jnp.ndarray
	reward: jnp.ndarray
	done: jnp.ndarray
	value: jnp.ndarray
	log_prob: jnp.ndarray
	info: jnp.ndarray


def make_train(config):
	num_updates = config.total_timesteps // config.num_steps // config.num_envs
	minibatch_size = config.num_envs * config.num_steps // config.num_minibatches

	env = Jax2DWrapper(config.env_path)
	env = EpisodeWrapper(env, config.episode_length, config.action_repeat)
	env = AutoResetWrapper(env)
	env = LogWrapper(env)
	env = ClipAction(env)
	env = VecEnv(env)

	def train(rng):
		# INIT NETWORK
		network = ActorCritic(env.action_space.shape[0])
		rng, _rng = jax.random.split(rng)
		init_x = {
			"vector": jnp.zeros(env.observation_space["vector"].shape),
			"image": jnp.zeros(env.observation_space["image"].shape),
		}
		network_params = network.init(_rng, init_x)
		tx = optax.chain(
			optax.clip_by_global_norm(config.max_grad_norm),
			optax.adam(config.learning_rate, eps=1e-5),
		)
		train_state = TrainState.create(
			apply_fn=network.apply,
			params=network_params,
			tx=tx,
		)

		# INIT ENV
		rng, _rng = jax.random.split(rng)
		reset_rng = jax.random.split(_rng, config.num_envs)
		env_state = env.reset(reset_rng)
		obsv = env_state.observation

		# TRAIN LOOP
		def _update_step(runner_state, _):
			# COLLECT TRAJECTORIES
			def _env_step(runner_state, _):
				train_state, env_state, last_obs, rng = runner_state

				# SELECT ACTION
				rng, _rng = jax.random.split(rng)
				pi, value = network.apply(train_state.params, last_obs)
				action = pi.sample(seed=_rng)
				log_prob = pi.log_prob(action)

				# STEP ENV
				rng, _rng = jax.random.split(rng)
				rng_step = jax.random.split(_rng, config.num_envs)
				env_state = env.step(rng_step, env_state, action)
				obsv, reward, done, info = (
					env_state.observation,
					env_state.reward,
					env_state.terminated,
					env_state.info,
				)
				transition = Transition(last_obs, action, reward, done, value, log_prob, info)
				runner_state = (train_state, env_state, obsv, rng)
				return runner_state, transition

			runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

			# CALCULATE ADVANTAGE
			train_state, env_state, last_obs, rng = runner_state
			_, last_val = network.apply(train_state.params, last_obs)

			def _calculate_gae(traj_batch, last_val):
				def _get_advantages(gae_and_next_value, transition):
					gae, next_value = gae_and_next_value
					done, value, reward = (
						transition.done,
						transition.value,
						transition.reward,
					)
					delta = reward + config.gamma * next_value * (1 - done) - value
					gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
					return (gae, value), gae

				_, advantages = jax.lax.scan(
					_get_advantages,
					(jnp.zeros_like(last_val), last_val),
					traj_batch,
					reverse=True,
					unroll=16,
				)
				return advantages, advantages + traj_batch.value

			advantages, targets = _calculate_gae(traj_batch, last_val)

			# UPDATE NETWORK
			def _update_epoch(update_state, _):
				def _update_minbatch(train_state, batch_info):
					traj_batch, advantages, targets = batch_info

					def _loss_fn(params, traj_batch, gae, targets):
						# RERUN NETWORK
						pi, value = network.apply(params, traj_batch.obs)
						log_prob = pi.log_prob(traj_batch.action)

						# CALCULATE VALUE LOSS
						value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
							-config.clip_eps, config.clip_eps
						)
						value_losses = jnp.square(value - targets)
						value_losses_clipped = jnp.square(value_pred_clipped - targets)
						value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

						# CALCULATE ACTOR LOSS
						ratio = jnp.exp(log_prob - traj_batch.log_prob)
						gae = (gae - gae.mean()) / (gae.std() + 1e-8)
						loss_actor1 = ratio * gae
						loss_actor2 = (
							jnp.clip(
								ratio,
								1.0 - config.clip_eps,
								1.0 + config.clip_eps,
							)
							* gae
						)
						loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
						loss_actor = loss_actor.mean()
						entropy = pi.entropy().mean()

						total_loss = (
							loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
						)
						return total_loss, (value_loss, loss_actor, entropy)

					grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
					total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
					train_state = train_state.apply_gradients(grads=grads)
					return train_state, total_loss

				train_state, traj_batch, advantages, targets, rng = update_state
				rng, _rng = jax.random.split(rng)
				batch_size = minibatch_size * config.num_minibatches
				assert (
					batch_size == config.num_steps * config.num_envs
				), "batch size must be equal to number of steps * number of envs"
				permutation = jax.random.permutation(_rng, batch_size)
				batch = (traj_batch, advantages, targets)
				batch = jax.tree_util.tree_map(
					lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
				)
				shuffled_batch = jax.tree_util.tree_map(
					lambda x: jnp.take(x, permutation, axis=0), batch
				)
				minibatches = jax.tree_util.tree_map(
					lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
					shuffled_batch,
				)
				train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
				update_state = (train_state, traj_batch, advantages, targets, rng)
				return update_state, total_loss

			update_state = (train_state, traj_batch, advantages, targets, rng)
			update_state, loss_info = jax.lax.scan(
				_update_epoch, update_state, None, config.update_epochs
			)
			train_state = update_state[0]
			metric = traj_batch.info
			rng = update_state[-1]

			if config.debug:

				def callback(metric):
					return_values = metric["returned_episode_returns"][-1]
					print(f"x/{int(num_updates)}: {jnp.mean(return_values)}")

				jax.debug.callback(callback, metric)

			runner_state = (train_state, env_state, last_obs, rng)
			return runner_state, ()

		rng, _rng = jax.random.split(rng)
		runner_state = (train_state, env_state, obsv, _rng)
		(train_state, env_state, obsv, _rng), _ = jax.lax.scan(_update_step, runner_state, None, num_updates)
		return train_state

	return train


def make_eval(config):
	env = Jax2DWrapper(config.env_path)
	env = EpisodeWrapper(env, config.episode_length, config.action_repeat)
	env = AutoResetWrapper(env)
	env = LogWrapper(env)
	env = ClipAction(env)
	env = VecEnv(env)

	def eval(rng, params):
		# Initialize network
		network = ActorCritic(env.action_space.shape[0])

		# Initialize environment
		rng, _rng = jax.random.split(rng)
		reset_rng = jax.random.split(_rng, config.num_eval_envs)
		env_state = env.reset(reset_rng)
		obsv = env_state.observation

		def _eval_step(carry, _):
			env_state, last_obs, rng = carry

			# Select action (deterministic)
			rng, _rng = jax.random.split(rng)
			pi, _ = network.apply(params, last_obs)
			action = pi.mode()

			# Step environment
			env_state = env.step(env_state, action)
			next_obs = env_state.observation

			carry = (env_state, next_obs, rng)
			return carry, env_state

		# Run evaluation episodes
		rng, _rng = jax.random.split(rng)
		_, env_states = jax.lax.scan(
			_eval_step, (env_state, obsv, _rng), None, config.episode_length
		)

		# Calculate mean return across episodes
		returns = env_states.info["returned_episode_returns"][-1]
		mean_return = jnp.mean(returns)

		return env_states, mean_return

	return eval
