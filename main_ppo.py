import hydra
from omegaconf import DictConfig

import jax

from ppo.ppo_continuous_action import make_train


@hydra.main(version_base=None, config_path="configs/ppo/", config_name="ppo_xs")
def main_ppo(config: DictConfig) -> None:
	key = jax.random.PRNGKey(config.seed)
	train_jit = jax.jit(make_train(config))
	out = train_jit(key)


if __name__ == "__main__":
	main_ppo()
