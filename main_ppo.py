import pickle
import hydra
from omegaconf import DictConfig

import jax

from ppo.ppo import make_train, make_eval


@hydra.main(version_base=None, config_path="configs/ppo/", config_name="ppo_xs")
def main_ppo(config: DictConfig) -> None:
	key = jax.random.PRNGKey(config.seed)

	train = jax.jit(make_train(config))
	train_state = train(key)

	eval = jax.jit(make_eval(config))
	env_state, mean_return = eval(key, train_state.params)
	print(f"mean return: {mean_return}")

	# Save params state
	with open("params.pickle", "wb") as f:
		pickle.dump(train_state.params, f)


if __name__ == "__main__":
	main_ppo()
