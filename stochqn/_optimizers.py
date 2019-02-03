import numpy as np, pandas as pd, warnings
from sklearn.model_selection import train_test_split
from stochqn._wrapper import oLBFGS_free, SQN_free, adaQN_free
from scipy.sparse import isspmatrix, isspmatrix_csr, vstack, csr_matrix

#### Pre-defined step size sequences
def _step_size_sqrt(initial_step_size, iteration_num):
	return initial_step_size / np.sqrt(iteration_num + 1)

def _step_size_const(initial_step_size, iteration_num):
	return initial_step_size

### Class with common methods
class _StochQN:
	def __init__(self):
		pass

	def _check_fit_inputs(self, X, y, sample_weight, additional_kwargs={}, check_sp=False):
		assert X.shape[0] > 0
		assert X.shape[0] == y.shape[0]
		if sample_weight is not None:
			assert sample_weight.shape[0] == X.shape[0]
		if additional_kwargs is None:
			additional_kwargs = dict()
		assert isinstance(additional_kwargs, dict)
		if check_sp:
			X, y = self._check_sp_type(X), self._check_sp_type(y)
			sample_weight = self._check_sp_type(sample_weight) if sample_weight is not None else None
		return X, y, sample_weight, additional_kwargs

	def _check_sp_type(self, X):
		if isspmatrix(X):
			if not isspmatrix_csr(X):
				warnings.warn("'.fit' method only supports sparse CSR matrices. Sparse inputs will be cast to CSR.")
				X = csr_matrix(X)
		return X

	def _get_long_batch(self, X, y, w, batch):
		### there will be a mismatch when running partial_fit before fit, or when running fit repeatedly
		### if it's with partial fit, it's possible to reuse the cases stored in the object for long batch
		diff = (batch + 1) % self.optimizer.bfgs_upd_freq

		if (batch + 1) >= (self.optimizer.bfgs_upd_freq - diff):
			st_ix = (batch + 1 - (self.optimizer.bfgs_upd_freq - diff)) * self.batch_size
			end_ix = min(X.shape[0], (batch + 1) * self.batch_size)
			X_long = X[st_ix : end_ix]
			y_long = y[st_ix : end_ix]
			w_long = w[st_ix : end_ix] if w is not None else None
		else:
			### In theory, one could take a chunk from the beginning and a chunk from the end,
			### but it's faster to just take a larger continuous sample.
			### Note that there is a check to ensure that the number of batches is >= upd_freq
			X_long = X[:min(X.shape[0], (self.optimizer.bfgs_upd_freq - diff) * self.batch_size)]
			y_long = y[:min(X.shape[0], (self.optimizer.bfgs_upd_freq - diff) * self.batch_size)]
			w_long = w[:min(X.shape[0], (self.optimizer.bfgs_upd_freq - diff) * self.batch_size)] if w is not None else None
		if diff > 0:
			self.stored_samples_X.append(X_long)
			self.stored_samples_y.append(y_long)
			self.stored_samples_w.append(w_long)
			X_long, y_long, w_long = self._get_stored_batch()

		return X_long, y_long, w_long

	def _stack_batch(self, lst_batch):
		nsparse = np.sum([isspmatrix(X) for X in lst_batch])
		if nsparse > 0:
			if nsparse < len(lst_batch):
				warnings.warn("When passing mixed batches of sparse and non-sparse data, these are forced to dense.")
				X_long = np.r_[tuple(lst_batch)]
			else:
				X_long = vstack(lst_batch)
		else:
			X_long = np.r_[tuple(lst_batch)]
		return X_long

	def _get_stored_batch(self):
		if len(self.stored_samples_X) == 0:
			raise ValueError("Unexpected error, please open an issue in GitHub explaining what you were doing.")
		X_long, y_long  = self._stack_batch(self.stored_samples_X), self._stack_batch(self.stored_samples_y)
		sum_w_missing = np.sum([w is None for w in self.stored_samples_w])
		if sum_w_missing == len(self.stored_samples_w):
			w_long = None
		else:
			if sum_w_missing != 0:
				warnings.warn("Passed batches with and without sample weights, missing weights will be set to 1.")
				self.stored_samples_w = [self.stored_samples_w[w] if self.stored_samples_w[w] is not None else np.ones((self.stored_samples_X[w].shape[0], 1)) for w in range(len(self.stored_samples_w))]
			w_long = self._stack_batch(self.stored_samples_w)

		self._reset_saved_batch()
		return X_long, y_long, w_long

	def _reset_saved_batch(self):
		self.stored_samples_X = list()
		self.stored_samples_y = list()
		self.stored_samples_w = list()

	def _add_common_attributes(self, x0, batches_per_epoch, step_size, grad_fun, obj_fun, pred_fun, decr_step_size,
			callback_epoch, callback_iter, valset_frac, tol, nepochs, kwargs_cb, random_state, shuffle_data,
			verbose, use_grad_diff):

		assert batches_per_epoch > 0
		assert isinstance(batches_per_epoch, int)
		
		assert step_size > 0
		if decr_step_size == "auto":
			decr_step_size = _step_size_sqrt
		elif decr_step_size is None:
			decr_step_size = _step_size_const
		else:
			if not callable(decr_step_size):
				raise ValueError("'decr_step_size' must be a function taking as input the initial step size and the iteration number, starting at zero.")
			decr_step_size = decr_step_size
		
		msg_cb = "Callback must be a function taking as argument the values of 'x' and additional keyword arguments, or 'None'"
		if callback_epoch is not None:
			if not callable(callback_epoch):
				raise ValueError(msg_cb)
		if callback_iter is not None:
			if not callable(callback_iter):
				raise ValueError(msg_cb)
		
		msg_fun = lambda fun_name, allow_None: "'" + fun_name + \
			"' must be a function that takes as argument the variables values, X, y, sample_weight, and additional keyword arguments" +\
			(", or 'None" if allow_None else "") + "."
		if not callable(grad_fun):
			raise ValueError(msg_fun("grad_fun", False))
		if pred_fun is not None:
			if not callable(pred_fun):
				raise ValueError(msg_fun("pred_fun", True))

		if valset_frac is not None:
			assert valset_frac > 0
			assert valset_frac < 1
			assert tol > 0
			if not callable(obj_fun):
				raise ValueError(msg_fun("obj_fun", False))
		assert nepochs > 0
		assert isinstance(nepochs, int)
		if kwargs_cb is not None:
			assert isinstance(kwargs_cb, dict)
		else:
			kwargs_cb = dict()

		if random_state is None:
			random_state = 1

		self.x = np.array(x0).astype('float64')
		self.n = self.x.shape[0]
		self.step_size = step_size
		self.obj_fun = obj_fun
		self.pred_fun = pred_fun
		self.grad_fun = grad_fun
		self.callback_epoch = callback_epoch
		self.callback_iter = callback_iter
		self.tol = tol
		self.nepochs = nepochs
		self.batches_per_epoch = batches_per_epoch
		self.decr_step_size = decr_step_size
		self.kwargs_cb = kwargs_cb
		self.valset_frac = valset_frac
		self.random_state = random_state
		self.verbose = bool(verbose)
		self.shuffle_data = bool(shuffle_data)
		self.use_grad_diff = bool(use_grad_diff)
		self.epoch = 0
		self.req = self.optimizer.run_optimizer(self.x, self.step_size)

		if self.optimizer_name != "oLBFGS":
			### in case partial fit is used
			self._reset_saved_batch()
		else:
			del self.use_grad_diff

	def fit(self, X, y, sample_weight=None, additional_kwargs={}, valset=None):
		"""
		Fit model to sample data

		Parameters
		----------
		X : array(n_samples, m)
			Sample data to which to fit the model.
		y : array(n_samples, )
			Labels or target values for the sample data.
		sample_weight : None or array(n_samples, )
			Observations weights for the sample data.
		additional_kwargs : dict
			Additional keyword arguments to pass to the objective, gradient, and Hessian-vector functions.
		valset : tuple(3)
			User-provided validation set containing (X_val, y_val, sample_weight_val).
			At the end of each epoch, will calculate objective function on this set, and if
			the decrease from the objective function in the previous epoch is below tolerance,
			will terminate procedure earlier.
			If 'valset_frac' was provided and a validation set is passed, 'valset_frac' will be ignored.
			Must provide objective function in order to use a validation set.

		Returns
		-------
		self : obj
			This object.
		"""
		X, y, sample_weight, additional_kwargs = self._check_fit_inputs(X, y, sample_weight, additional_kwargs, check_sp=True)
		if valset is not None:
			if self.obj_fun is None:
				raise ValueError("Must provide objective function when using a validation set for monitoring.")
			assert isinstance(valset, tuple)
			assert len(valset) == 3
			X_val, y_val, w_val = valset
			X_val, y_val, w_val, additional_kwargs = self._check_fit_inputs(X_val, y_val, w_val, additional_kwargs, check_sp=False)
			if self.valset_frac is not None:
				warnings.warn("'valset_frac' is ignored when passign a validation set to '.fit'.")

		elif self.valset_frac is not None:
			if sample_weight is None:
				X, X_val, y, y_val = train_test_split(X, y, test_size=self.valset_frac, random_state=self.random_state)
				w_val = None
			else:
				X, X_val, y, y_val, sample_weight, w_val = train_test_split(X, y, sample_weight, test_size=self.valset_frac, random_state=self.random_state)
		else:
			X_val, y_val, w_val = None, None, None
		obj_last_epoch = np.inf
		
		print_term_msg = True if self.verbose else False

		self.batch_size = int(np.ceil(X.shape[0] / self.batches_per_epoch))
		for self.epoch in range(self.nepochs):
			if self.shuffle_data:
				np.random.seed(self.random_state + self.epoch)
				rand_ord = np.argsort( np.random.random(size=X.shape[0]) )
				X = X[rand_ord]
				y = y[rand_ord]
				sample_weight = sample_weight[rand_ord] if sample_weight is not None else None

			for batch in range(self.batches_per_epoch):
				st_batch_ix = batch * self.batch_size
				end_batch_ix = min(X.shape[0], (batch + 1) * self.batch_size)
				X_batch = X[st_batch_ix : end_batch_ix]
				y_batch = y[st_batch_ix : end_batch_ix]
				w_batch = sample_weight[st_batch_ix : end_batch_ix] if sample_weight is not None else None

				self._fit_batch(X_batch, y_batch, w_batch, additional_kwargs, is_user_batch=False,
					X_full=X, y_full=y, w_full=sample_weight, X_val=X_val, y_val=y_val, w_val=w_val, batch=batch)

			if self.callback_epoch is not None:
				self.callback_epoch(self.x, **self.kwargs_cb)

			if X_val is not None and self.obj_fun is not None:
				obj_this_epoch = self.obj_fun(self.x, X_val, y_val, sample_weight=w_val, **additional_kwargs)
				if self.verbose:
					print((self.optimizer_name + " - epoch: %2d, f(x): %12.4f") % (self.epoch + 1, obj_this_epoch) )
				if (obj_last_epoch - obj_this_epoch) < self.tol and obj_this_epoch <= obj_last_epoch:
					if self.verbose:
						print(self.optimizer_name + " - Optimization procedure terminated (decrease below tolerance).")
						print_term_msg = False
					break
				else:
					obj_last_epoch = obj_this_epoch

		if print_term_msg:
			print(self.optimizer_name + " - Optimization procedure terminated (reached number of epochs).")

		return self

	def partial_fit(self, X, y, sample_weight=None, additional_kwargs={}):
		"""
		Update model with user-provided batches of data

		Note
		----
		In SQN and adaQN, the data passed to all calls in partial fit will be stored in a limited-memory
		container which will be used to calculate Hessian-vector products or large-batch gradients.
		The size of this container is determined by the inputs 'batch_size' and 'bfgs_upd_freq' passed
		in the constructor call.

		Note
		----
		The step size in partial fit is determined by the number of optimizer iterations rather than the number
		of epochs, thus for a given amount of data, the default step size will be much smaller than when calling 'fit'.
		Recommended to provide a custom step size function ('decr_step_size' in the initialization), as otherwise the
		step size sequence will be too small.

		Parameters
		----------
		X : array(n_samples, m)
			Sample data to with which to update the model.
		y : array(n_samples, )
			Labels or target values for the sample data.
		sample_weight : None or array(n_samples, )
			Observations weights for the sample data.
		additional_kwargs : dict
			Additional keyword arguments to pass to the objective, gradient, and Hessian-vector functions.

		Returns
		-------
		self : obj
			This object.
		"""
		X, y, sample_weight, additional_kwargs = self._check_fit_inputs(X, y, sample_weight, additional_kwargs, check_sp=False)

		save_batch = False
		if self.optimizer_name == "SQN":
			save_batch = True
		elif self.optimizer_name == "adaQN":
			if self.use_grad_diff or (self.optimizer.max_incr > 0 if self.optimizer.max_incr is not None else False):
				save_batch = True

		if save_batch:
			self.stored_samples_X.append(X)
			self.stored_samples_y.append(y)
			self.stored_samples_w.append(sample_weight)

		self._fit_batch(X, y, sample_weight, additional_kwargs, is_user_batch=True)
		return self

	def _fit_batch(self, X_batch, y_batch, w_batch, additional_kwargs, is_user_batch=False,
		X_full=None, y_full=None, w_full=None, X_val=None, y_val=None, w_val=None, batch=None):
		
		while True:
			if self.req["task"] == "calc_grad" or self.req["task"] == "calc_grad_same_batch":
				self.optimizer.update_gradient(self.grad_fun(self.req["requested_on"], X_batch, y_batch, sample_weight=w_batch, **additional_kwargs))
			
			else:
				if self.req["task"] == "calc_fun_val_batch" and X_val is not None:
					self.optimizer.update_function(self.obj_fun(self.req["requested_on"], X_val, y_val, sample_weight=w_val, **additional_kwargs))
				
				else:
					if is_user_batch:
						X_long, y_long, w_long = self._get_stored_batch()
					else:
						X_long, y_long, w_long = self._get_long_batch(X_full, y_full, w_full, batch)

					if self.req["task"] == "calc_grad_big_batch":
						self.optimizer.update_gradient(self.grad_fun(self.req["requested_on"], X_long, y_long, sample_weight=w_long, **additional_kwargs))
					elif self.req["task"] == "calc_hess_vec":
						self.optimizer.update_hess_vec(self.hess_vec_fun(self.req["requested_on"][0], self.req["requested_on"][1], X_long, y_long, sample_weight=w_long, **additional_kwargs))
					elif self.req["task"] == "calc_fun_val_batch":
						self.optimizer.update_function(self.obj_fun(self.req["requested_on"], X_long, y_long, sample_weight=w_long, **additional_kwargs))
					else:
						raise ValueError("Unexpected error. Please open an issue in GitHub explaining what you were doing.")

			if is_user_batch:
				step_size = self.decr_step_size(self.step_size, self.niter)
			else:
				step_size = self.decr_step_size(self.step_size, self.epoch)
				
			self.req = self.optimizer.run_optimizer(self.x, step_size)
			
			if self.verbose:
				if self.req["info"]["iteration_info"] != "no_problems_encountered":
					if is_user_batch:
						print( (self.optimizer_name + " - at iteration %3d: " + self.req["info"]["iteration_info"]) % self.niter )
					else:
						print( (self.optimizer_name + " - at iteration %3d, epoch %2d: " + self.req["info"]["iteration_info"]) % (self.niter, self.epoch + 1) )
			
			if self.req["task"] == "calc_grad":
				if self.callback_iter is not None:
					self.callback_iter(self.x, **self.kwargs_cb)
				break

	def predict(self, X, additional_kwargs={}):
		"""
		Make predictions on new data

		Note
		----
		Using this method requires passing 'pred_fun' in the initialization.

		Parameters
		----------
		X : array(n_samples, m)
			New data to pass to user-provided predict function.
		additional_kwargs : dict
			Additional keyword arguments to pass to user-provided predict function.
		"""
		if self.pred_fun is None:
			raise ValueError("Must supply predict function in order to call this method.")
		else:
			return self.pred_fun(self.x, X, **additional_kwargs)

	def get_x(self):
		"""
		Get a copy of current values of the variables

		Returns
		-------
		x : array(n, )
			Current variable values.
		"""
		return self.x.copy()

#### Optimizers
class oLBFGS(_StochQN):
	def __init__(self, x0, grad_fun, obj_fun=None, pred_fun=None, batches_per_epoch=25, step_size=1e-3, decr_step_size="auto",
		shuffle_data=True, random_state=1, nepochs=25, valset_frac=None, tol=1e-1,
		callback_epoch=None, callback_iter=None, kwargs_cb={}, verbose=True,
		mem_size=10, hess_init=None, min_curvature=1e-4, y_reg=None, check_nan=True, nthreads=-1):
		"""
		oLBFGS optimizer

		Optimizes an empirical (convex) loss function over batches of sample data.

		Parameters
		----------
		x0 : array (m, )
			Initial values of the variables to optimize (refered hereafter as 'x').
		grad_fun : function(x, X, y, sample_weight, **kwargs) --> array(m, )
			Function that calculates the empirical gradient at values 'x' on data 'X' and 'y'.
			Note: output must be one-dimensional and with the same number of entries as 'x',
			otherwise the Python session might segfault.
			(The extra keyword arguments are passed in the 'fit' method, not here)
		obj_fun : function(x, X, y, sample_weight, **kwargs) --> float
			Function that calculates the empirical objective value at values 'x' on data 'X' and 'y'.
			Only used when using a validation set ('valset_frac' not None, or 'valset' passed to fit).
			Ignored when fitting the data in user-provided batches.
			(The extra keyword arguments are passed in the 'fit' method, not here)
		pred_fun : None or function(xopt, X)
			Prediction function taking as input the optimal 'x' values as obtained by the
			optimization procedure, and new observation 'X' on which to make predictions.
			If passed, will have an additional method oLBFGS.predict(X, *args) that calls
			this function with current values of 'x'.
		batches_per_epoch : int
			Number of batches per epoch (each batch will have the same number of observations except for the
			last one which might be smaller).
		step_size : float
			Initial step size to use.
			(Can be modified after object is already initialized)
		decr_step_size : str "auto", None, or function(initial_step_size, epoch) -> float
			Function that determines the step size during each epoch, taking as input the initial
			step size and the epoch number (starting at zero).
			If "auto", will use 1/sqrt(iteration).
			If None, will use constant step size.
			For 'partial_fit', it will take as input the number of iterations of the algorithm rather than epoch,
			so it's very recommended to provide a custom function when passing data in user-provided batches.
			Can be modified after the object has been initialized (oLBFGS.decr_step_size)
		shuffle_data : bool
			Whether to shuffle the data at the beginning of each epoch.
		random_state : int
			Random seed to use for shuffling data and selecting validation set.
			The algorithm is deterministic so it's not used for anything else.
		nepochs : int
			Number of epochs for which to run the optimization procedure.
			Might terminate earlier if using a validation set for monitoring.
		valset_frac : float(0, 1) or None
			Percent of the data to use as validation set for early stopping.
			Can also pass a user-provided validation set to 'fit', in which case it will
			be ignored.
			If passing None, will run for the number of epochs passed in 'nepochs'.
		tol : float
			If the objective function calculated on the validation set decrease by less than
			'tol' upon completion of an epoch, will terminate the optimization procedure.
			Ignored when not using a validation set.
		callback_epoch : None or function*(x, **kwargs)
			Callback function to call at the end of each epoch
		callback_iter : None or function*(x, **kwargs)
			Callback function to call at the end of each iteration
		kwargs_cb : tuple
			Additional arguments to pass to 'callback' and 'stop_crit'.
			(Can be modified after object is already initialized)
		verbose : bool
			Whether to print messages when there is some problem during an iteration
			(e.g. correction pair not meeting minum curvature).
		mem_size : int
			Number of correction pairs to store for approximation of Hessian-vector products.
		hess_init : float or None
			value to which to initialize the diagonal of H0.
			If passing 0, will use the same initializion as for SQN (s_last*y_last / y_last*y_last).
		min_curvature : float or None
			Minimum value of s*y / s*s in order to accept a correction pair.
		y_reg : float or None
			regularizer for 'y' vector (gets added y_reg * s)
		check_nan : bool
			Whether to check for variables becoming NaN after each iteration, and reverting the step if they do
			(will also reset BFGS memory).
		nthreads : int
			Number of parallel threads to use. If set to -1, will determine the number of available threads and use
			all of them. Note however that not all the computations can be parallelized.

		References
		----------
		Schraudolph, N.N., Yu, J. and Günter, S., 2007, March.
		"A stochastic quasi-Newton method for online convex optimization."
		In Artificial Intelligence and Statistics (pp. 436-443).
		"""

		self.optimizer_name = "oLBFGS"
		self.optimizer = oLBFGS_free(mem_size, hess_init, min_curvature, y_reg, check_nan, nthreads)
		use_grad_diff = True
		self._add_common_attributes(x0, batches_per_epoch, step_size, grad_fun, obj_fun, pred_fun, decr_step_size,
			callback_epoch, callback_iter, valset_frac, tol, nepochs, kwargs_cb, random_state, shuffle_data,
			verbose, use_grad_diff)

	@property
	def niter(self):
		return self.optimizer._oLBFGS.niter

class SQN(_StochQN):
	def __init__(self, x0, grad_fun, obj_fun=None, hess_vec_fun=None, pred_fun=None, batches_per_epoch=25, step_size=1e-3, decr_step_size="auto",
		shuffle_data=True, random_state=1, nepochs=25, valset_frac=None, tol=1e-1,
		callback_epoch=None, callback_iter=None, kwargs_cb={}, verbose=True,
		mem_size=10, bfgs_upd_freq=20, min_curvature=1e-4, y_reg=None, use_grad_diff=False, check_nan=True, nthreads=-1):
		"""
		SQN optimizer

		Optimizes an empirical (convex) loss function over batches of sample data.

		Parameters
		----------
		x0 : array (m, )
			Initial values of the variables to optimize (refered hereafter as 'x').
		grad_fun : function(x, X, y, sample_weight, **kwargs) --> array(m, )
			Function that calculates the empirical gradient at values 'x' on data 'X' and 'y'.
			Note: output must be one-dimensional and with the same number of entries as 'x',
			otherwise the Python session might segfault.
			(The extra keyword arguments are passed in the 'fit' method, not here)
		obj_fun : function(x, X, y, sample_weight, **kwargs) --> float
			Function that calculates the empirical objective value at values 'x' on data 'X' and 'y'.
			Only used when using a validation set ('valset_frac' not None, or 'valset' passed to fit).
			Ignored when fitting the data in user-provided batches.
			(The extra keyword arguments are passed in the 'fit' method, not here)
		hess_vec_fun : function(x, vec, X, y, sample_weight, **kwargs) --> array(m, )
			Function that calculates the product of a vector the empirical Hessian at values 'x' on data 'X' and 'y'.
			Ignored when using 'use_grad_diff=True'.
			Note: output must be one-dimensional and with the same number of entries as 'x',
			otherwise the Python session might segfault.
			These products are calculated on a larger batch than the gradients (given by batch_size * bfgs_upd_freq).
			(The extra keyword arguments are passed in the 'fit' method, not here)
		pred_fun : None or function(xopt, X)
			Prediction function taking as input the optimal 'x' values as obtained by the
			optimization procedure, and new observation 'X' on which to make predictions.
			If passed, will have an additional method oLBFGS.predict(X, *args) that calls
			this function with current values of 'x'.
		batches_per_epoch : int
			Number of batches per epoch (each batch will have the same number of observations except for the
			last one which might be smaller).
		step_size : float
			Initial step size to use.
			(Can be modified after object is already initialized)
		decr_step_size : str "auto", None, or function(initial_step_size, epoch) -> float
			Function that determines the step size during each epoch, taking as input the initial
			step size and the epoch number (starting at zero).
			If "auto", will use 1/sqrt(iteration).
			If None, will use constant step size.
			For 'partial_fit', it will take as input the number of iterations of the algorithm rather than epoch,
			so it's very recommended to provide a custom function when passing data in user-provided batches.
			Can be modified after the object has been initialized (oLBFGS.decr_step_size)
		shuffle_data : bool
			Whether to shuffle the data at the beginning of each epoch.
		random_state : int
			Random seed to use for shuffling data and selecting validation set.
			The algorithm is deterministic so it's not used for anything else.
		nepochs : int
			Number of epochs for which to run the optimization procedure.
			Might terminate earlier if using a validation set for monitoring.
		valset_frac : float(0, 1) or None
			Percent of the data to use as validation set for early stopping.
			Can also pass a user-provided validation set to 'fit', in which case it will
			be ignored.
			If passing None, will run for the number of epochs passed in 'nepochs'.
		tol : float
			If the objective function calculated on the validation set decrease by less than
			'tol' upon completion of an epoch, will terminate the optimization procedure.
			Ignored when not using a validation set.
		callback_epoch : None or function*(x, **kwargs)
			Callback function to call at the end of each epoch
		callback_iter : None or function*(x, **kwargs)
			Callback function to call at the end of each iteration
		kwargs_cb : tuple
			Additional arguments to pass to 'callback' and 'stop_crit'.
			(Can be modified after object is already initialized)
		verbose : bool
			Whether to print messages when there is some problem during an iteration
			(e.g. correction pair not meeting minum curvature).
		mem_size : int
			Number of correction pairs to store for approximation of Hessian-vector products.
		bfgs_upd_freq : int
			Number of iterations (batches) after which to generate a BFGS correction pair.
		min_curvature : float or None
			Minimum value of s*y / s*s in order to accept a correction pair.
		y_reg : float or None
			regularizer for 'y' vector (gets added y_reg * s)
		use_grad_diff : bool
			Whether to create the correction pairs using differences between gradients instead of Hessian-vector products.
			These gradients are calculated on a larger batch than the regular ones (given by batch_size * bfgs_upd_freq).
		check_nan : bool
			Whether to check for variables becoming NaN after each iteration, and reverting the step if they do
			(will also reset BFGS memory).
		nthreads : int
			Number of parallel threads to use. If set to -1, will determine the number of available threads and use
			all of them. Note however that not all the computations can be parallelized.

		References
		----------
		[1] Byrd, R.H., Hansen, S.L., Nocedal, J. and Singer, Y., 2016. "A stochastic quasi-Newton method for large-scale optimization."
			SIAM Journal on Optimization, 26(2), pp.1008-1031.
		[2] Wright, S. and Nocedal, J., 1999. "Numerical optimization." (ch 7)
			Springer Science, 35(67-68), p.7.
		"""

		if not use_grad_diff and hess_vec_fun is None:
			raise ValueError("If not using 'use_grad_diff', must provide function that evaluates Hessian-vector product.")
		if hess_vec_fun is not None:
			if use_grad_diff:
				warnings.warn("Hessian-vector function is ignored when passing 'use_grad_diff=True'.")
				use_grad_diff = None
			else:
				if not callable(hess_vec_fun):
					raise ValueError("'hess_vec_fun' must be a function that takes as input the values of 'x' and a vector, returning the Hessian-vector product.")

		self.optimizer_name = "SQN"
		self.optimizer = SQN_free(mem_size, bfgs_upd_freq, min_curvature, y_reg, use_grad_diff, check_nan, nthreads)
		self._add_common_attributes(x0, batches_per_epoch, step_size, grad_fun, obj_fun, pred_fun, decr_step_size,
			callback_epoch, callback_iter, valset_frac, tol, nepochs, kwargs_cb, random_state, shuffle_data,
			verbose, use_grad_diff)
		self.hess_vec_fun = hess_vec_fun

	@property
	def niter(self):
		return self.optimizer._SQN.niter

class adaQN(_StochQN):
	def __init__(self, x0, grad_fun, obj_fun=None, pred_fun=None, batches_per_epoch=25, step_size=1e-1, decr_step_size=None,
		shuffle_data=True, random_state=1, nepochs=25, valset_frac=None, tol=1e-1,
		callback_epoch=None, callback_iter=None, kwargs_cb={}, verbose=True,
		mem_size=10, fisher_size=100, bfgs_upd_freq=20, max_incr=1.01, min_curvature=1e-4, y_reg=None,
		scal_reg=1e-4, rmsprop_weight=None, use_grad_diff=False, check_nan=True, nthreads=-1):
		"""
		adaQN optimizer

		Optimizes an empirical (possibly non-convex) loss function over batches of sample data.

		Parameters
		----------
		x0 : array (m, )
			Initial values of the variables to optimize (refered hereafter as 'x').
		grad_fun : function(x, X, y, sample_weight, **kwargs) --> array(m, )
			Function that calculates the empirical gradient at values 'x' on data 'X' and 'y'.
			Note: output must be one-dimensional and with the same number of entries as 'x',
			otherwise the Python session might segfault.
			(The extra keyword arguments are passed in the 'fit' method, not here)
		obj_fun : function(x, X, y, sample_weight, **kwargs) --> float
			Function that calculates the empirical objective value at values 'x' on data 'X' and 'y'.
			Will be ignored if passing 'max_incr=None' and no validation set
			('valset_frac=None', and no 'valset' passed to fit).
			(The extra keyword arguments are passed in the 'fit' method, not here)
		pred_fun : None or function(xopt, X)
			Prediction function taking as input the optimal 'x' values as obtained by the
			optimization procedure, and new observation 'X' on which to make predictions.
			If passed, will have an additional method oLBFGS.predict(X, *args) that calls
			this function with current values of 'x'.
		batches_per_epoch : int
			Number of batches per epoch (each batch will have the same number of observations except for the
			last one which might be smaller).
		step_size : float
			Initial step size to use.
			(Can be modified after object is already initialized)
		decr_step_size : str "auto", None, or function(initial_step_size, epoch) -> float
			Function that determines the step size during each epoch, taking as input the initial
			step size and the epoch number (starting at zero).
			If "auto", will use 1/sqrt(iteration).
			If None, will use constant step size.
			For 'partial_fit', it will take as input the number of iterations of the algorithm rather than epoch,
			so it's very recommended to provide a custom function when passing data in user-provided batches.
			Can be modified after the object has been initialized (oLBFGS.decr_step_size)
		shuffle_data : bool
			Whether to shuffle the data at the beginning of each epoch.
		random_state : int
			Random seed to use for shuffling data and selecting validation set.
			The algorithm is deterministic so it's not used for anything else.
		nepochs : int
			Number of epochs for which to run the optimization procedure.
			Might terminate earlier if using a validation set for monitoring.
		valset_frac : float(0, 1) or None
			Percent of the data to use as validation set for early stopping.
			Can also pass a user-provided validation set to 'fit', in which case it will
			be ignored.
			If passing None, will run for the number of epochs passed in 'nepochs'.
		tol : float
			If the objective function calculated on the validation set decrease by less than
			'tol' upon completion of an epoch, will terminate the optimization procedure.
			Ignored when not using a validation set.
		callback_epoch : None or function*(x, **kwargs)
			Callback function to call at the end of each epoch
		callback_iter : None or function*(x, **kwargs)
			Callback function to call at the end of each iteration
		kwargs_cb : tuple
			Additional arguments to pass to 'callback' and 'stop_crit'.
			(Can be modified after object is already initialized)
		verbose : bool
			Whether to print messages when there is some problem during an iteration
			(e.g. correction pair not meeting minum curvature).
		mem_size : int
			Number of correction pairs to store for approximation of Hessian-vector products.
		fisher_size : int or None
			Number of gradients to store for calculation of the empirical Fisher product with gradients.
			If passing 'None', will force 'use_grad_diff' to 'True'.
		bfgs_upd_freq : int
			Number of iterations (batches) after which to generate a BFGS correction pair.
		max_incr : float or None
			Maximum ratio of function values in the validation set under the average values of 'x' during current epoch
			vs. previous epoch. If the ratio is above this threshold, the BFGS and Fisher memories will be reset, and 'x'
			values reverted to their previous average.
			If not using a validation set, will take a longer batch for function evaluations (same as used for gradients
			when using 'use_grad_diff=True').
		min_curvature : float or None
			Minimum value of s*y / s*s in order to accept a correction pair.
		y_reg : float or None
			regularizer for 'y' vector (gets added y_reg * s)
		scal_reg : float
			Regularization parameter to use in the denominator for AdaGrad and RMSProp scaling.
		rmsprop_weight : float(0,1) or None
			If not 'None', will use RMSProp formula instead of AdaGrad for approximated inverse-Hessian initialization.
			(Recommended to use lower initial step size + passing 'decr_step_size')
		use_grad_diff : bool
			Whether to create the correction pairs using differences between gradients instead of Fisher matrix.
			These gradients are calculated on a larger batch than the regular ones (given by batch_size * bfgs_upd_freq).
			If 'True', fisher_size will be set to None, and empirical Fisher matrix will not be used.
		check_nan : bool
			Whether to check for variables becoming NaN after each iteration, and reverting the step if they do
			(will also reset BFGS memory).
		nthreads : int
			Number of parallel threads to use. If set to -1, will determine the number of available threads and use
			all of them. Note however that not all the computations can be parallelized.

		References
		----------
		[1] Keskar, N.S. and Berahas, A.S., 2016, September. "adaQN: An Adaptive Quasi-Newton Algorithm for Training RNNs."
			In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 1-16). Springer, Cham.
		[2] Wright, S. and Nocedal, J., 1999. "Numerical optimization." (ch 7)
			Springer Science, 35(67-68), p.7.
		"""

		if max_incr is not None:
			if obj_fun is None:
				raise ValueError("Must provide objective function when passing 'max_incr'.")
		if use_grad_diff and fisher_size is not None:
			warnings.warn("'fisher_size' ignored when using 'use_grad_diff=True'.")
		if fisher_size is None:
			use_grad_diff = True

		self.optimizer_name = "adaQN"
		self.optimizer = adaQN_free(mem_size, fisher_size, bfgs_upd_freq, max_incr, min_curvature, scal_reg,
			rmsprop_weight, y_reg, use_grad_diff, check_nan, nthreads)
		self._add_common_attributes(x0, batches_per_epoch, step_size, grad_fun, obj_fun, pred_fun, decr_step_size,
			callback_epoch, callback_iter, valset_frac, tol, nepochs, kwargs_cb, random_state, shuffle_data,
			verbose, use_grad_diff)

	@property
	def niter(self):
		return self.optimizer._adaQN.niter
 
