import numpy as np, tensorflow as tf
from stochqn._optimizers import oLBFGS, adaQN
from copy import deepcopy
import ctypes, warnings

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
		self.pass_args = optimizer_kwargs
		self.optimizer = None
		tf.contrib.opt.ExternalOptimizerInterface.__init__(self, loss, var_list, None, None, None)

	def _minimize(self, initial_val, loss_grad_func, equality_funcs,
				  equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
				  packed_bounds, step_callback, optimizer_kwargs):

		def grad_fun(*args, **kwargs):
			return loss_grad_func(self.optimizer.x)[1]
		def obj_fun(*args, **kwargs):
			return loss_grad_func(self.optimizer.x)[0]

		if initial_val.dtype != ctypes.c_double:
			warnings.warn("Passed a variable with dtype different from C double (float 64) - will copy and cast the data.")
			correct_dtype = False
			initial_val_pass = initial_val.astype(ctypes.c_double)
		else:
			correct_dtype = True
			initial_val_pass = initial_val

		if self.optimizer is None:
			self.pass_args["valset_frac"] = None
			if self.optimizer_name == "adaQN":
				self.pass_args["max_incr"] = None
			if self.optimizer_name == "oLBFGS":
				self.optimizer = oLBFGS(initial_val_pass, grad_fun=grad_fun, obj_fun=obj_fun, **self.pass_args)
			elif self.optimizer_name == "adaQN":
				self.optimizer = adaQN(initial_val_pass, grad_fun=grad_fun, obj_fun=obj_fun, **self.pass_args)
			else:
				raise ValueError("'optimizer' must be one of 'oLBFGS' or 'adaQN'.")
		self.optimizer.partial_fit(_Subscriptable_None(initial_val_pass.shape[0]), _Subscriptable_None(initial_val_pass.shape[0]))

		if correct_dtype:
			return self.optimizer.x
		else:
			initial_val[:] = self.optimizer.x
			return self.optimizer.x.astype(initial_val.dtype)
