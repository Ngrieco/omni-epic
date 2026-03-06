import gymnasium as gym


class VisionWrapper(gym.Wrapper):

	def __init__(self, env, height=64, width=64, use_depth=True, fov=90.):
		super().__init__(env)
		self._height = height
		self._width = width
		self._use_depth = use_depth
		self._fov = fov

	def vision(self):
		return self.env.robot.vision(self._height, self._width, self._use_depth, self._fov)
	
	def get_success(self):
		return self.env.get_success()
	
	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs)
	
	def render3p(self, *args, **kwargs):
		return self.env.render3p(*args, **kwargs)
	
	def close(self):
		return self.env.close()
