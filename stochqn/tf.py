import numpy as np, tensorflow as tf
from stochqn._optimizers import oLBFGS, adaQN

#### Workaround for passing data in Tensorflow
class _Subscriptable_None:
	def __init__(self, shape):
		self.shape = (shape,)

	def __getitem__(self, item):
		if item.__class__.__name__ == "ndarray":
			self.shape = (item.shape[0], )
		else:
			self.shape = (item.stop - item.start, )
		return self

class TensorflowStochQNOptimizer(tf.contrib.opt.ExternalOptimizerInterface):
	def __init__(self, loss, var_list=None, optimizer="adaQN", **optimizer_kwargs):
		"""
		StochQN optimizer for Tensorflow

		Parameters
		----------
		loss : scalar `Tensor`
			Objective to minimize.
		var_list : list or None
			Optional `list` of `Variable` objects to update to minimize
			`loss`. Defaults to the list of variables collected in the graph
			under the key `GraphKeys.TRAINABLE_VARIABLES`.
		optimizer : str, one of 'oLBFGS', 'adaQN'
			Optimizer to use
		user_defined_batches : bool
			Whether the data will be passed in batches to which 'partial_fit' will be called,
			or all in one go, in which case the optimizer will define the batches.
			Recommended to pass a custom step size function if passing data in batches.
		optimizer_kwargs : dict, optional
			Additional options to pass to the optimizer (see each optimizers documentation for details).
			Note that it should not contain the options 'valset_frac' or 'max_incr', which will be forced
			to 'None'.
		"""
		if optimizer not in ["oLBFGS", "adaQN"]:
			raise ValueError("'optimizer' must be one of 'oLBFGS' or 'adaQN'.")

		self.optimizer_name = optimizer
		self.optimizer_kwargs = optimizer_kwargs
		self.optimizer = None
		tf.contrib.opt.ExternalOptimizerInterface.__init__(self, loss, var_list, None, None, None)

	def _minimize(self, initial_val, loss_grad_func, equality_funcs,
		equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
		packed_bounds, step_callback, optimizer_kwargs):
		def grad_fun(x, y, *args, **kwargs):
			return loss_grad_func(x)[1]
		def obj_fun(x, y, *args, **kwargs):
			return loss_grad_func(x)[0]
		if self.optimizer is None:
			self.optimizer_kwargs["valset_frac"] = None
			if self.optimizer_name == "adaQN":
				self.optimizer_kwargs["max_incr"] = None
			if self.optimizer_name == "oLBFGS":
				self.optimizer = oLBFGS(initial_val, grad_fun=grad_fun, obj_fun=obj_fun, **self.optimizer_kwargs)
			elif self.optimizer_name == "adaQN":
				self.optimizer = adaQN(initial_val, grad_fun=grad_fun, obj_fun=obj_fun, **self.optimizer_kwargs)
			else:
				raise ValueError("'optimizer' must be one of 'oLBFGS' or 'adaQN'.")
		self.optimizer.partial_fit(initial_val, _Subscriptable_None(initial_val.shape[0]))
		return self.optimizer.x
