import numpy as np, warnings
from stochqn._optimizers import oLBFGS, SQN, adaQN, _StochQN
from sklearn.linear_model.logistic import _logistic_loss_and_grad, _logistic_grad_hess
from sklearn.linear_model.logistic import _multinomial_loss_grad, _multinomial_grad_hess
from scipy.sparse import isspmatrix

def _grad_fun_multi(w, X, y, sample_weights=None, reg_param=0):
	return _multinomial_loss_grad(w, X, y, reg_param, sample_weights)[1]
def _obj_fun_mult(w, X, y, sample_weights=None, reg_param=0):
	return _multinomial_loss_grad(w, X, y, reg_param, sample_weights)[0]
def _hessvec_fun_mult(w, v, X, y, sample_weights=None, reg_param=0):
	temp = _multinomial_grad_hess(w, X, y, reg_param, sample_weights)[1]
	return temp(v)
def _pred_fun_mult(w, X, nclasses):
	w = w.reshape((nclasses, -1))
	if w.shape[1] == X.shape[1]:
		pred = w.dot(X.T)
	else:
		pred = X.dot(w[:, :X.shape[1]].T) + w[:, -1].reshape((1, -1))
	return 1 / (1 + np.exp(-pred))

def _grad_fun_bin(w, X, y, sample_weights=None, reg_param=0):
	return _logistic_loss_and_grad(w, X, y, reg_param)[1]
def _hessvec_fun_bin(w, v, X, y, sample_weights=None, reg_param=0):
	temp = _logistic_grad_hess(w, X, y, reg_param)[1]
	return temp(v)
def _obj_fun_bin(w, X, y, sample_weights=None, reg_param=0):
	return _logistic_loss_and_grad(w, X, y, reg_param)[0]
def _pred_fun_bin(w, X):
	if w.shape[0] == X.shape[1]:
		pred = X.dot(w)
	else:
		pred = X.dot(w[:X.shape[1]]) + w[-1]
	return (1 / (1 + np.exp(-pred))).reshape(-1)

class StochasticLogisticRegression:
	def __init__(self, reg_param=1e-3, fit_intercept=True, random_state=1, optimizer="SQN", step_size=1e-1, valset_frac=0.1, verbose=False, **optimizer_kwargs):
		"""
		Logistic Regression fit with stochastic quasi-Newton optimizer

		Parameters
		----------
		reg_param : float
			Strength of l2 regularization. Note that the loss function has an average log-loss over observations,
			so the optimal regulatization will likely be a lot smaller than for scikit-learn's (which uses sum instead).
		step_size : float
			Initial step size to use. Note that it will be decreased after each epoch when using 'fit',
			but will not be decreased after calling 'partial_fit'.
		fit_intercept : bool
			Whether to add an intercept to the model parameters.
		random_state : int
			Random seed to use.
		optimizer : str, one of 'oLBFGS', 'SQN', 'adaQN'
			Optimizer to use.
		optimizer_kwargs : dict, optional
			Additional options to pass to the optimizer (see each optimizer's documentation).
		"""
		assert optimizer in ["oLBFGS", "SQN", "adaQN"]
		assert step_size > 0
		assert isinstance(step_size, float)
		assert reg_param >= 0
		assert isinstance(reg_param, float)
		optimizer_kwargs["step_size"] = step_size
		optimizer_kwargs["valset_frac"] = valset_frac
		optimizer_kwargs["verbose"] = verbose

		self.optimizer_name = optimizer
		self.optimizer = None
		self.optimizer_kwargs = optimizer_kwargs
		self.reg_param = reg_param
		self.nclasses = None
		self._is_mult = None
		self.fit_intercept = bool(fit_intercept)
		self.is_fitted = False
		self.random_state = random_state

	@property
	def coef_(self):
		if not self.is_fitted:
			return None
		if self._is_mult:
			if self.fit_intercept:
				return (self.optimizer.x.reshape((self.nclasses, -1)))[:, -1 + self.optimizer.x.shape[0] / self.nclasses]
			else:
				return self.optimizer.x.reshape((self.nclasses, -1))
		else:
			if self.fit_intercept:
				return self.optimizer.x[:self.optimizer.x.shape[0] - 1]
			else:
				return self.optimizer.x

	@property
	def intercept_(self):
		if not self.is_fitted:
			return None
		if self._is_mult:
			if self.fit_intercept:
				return (self.optimizer.xreshape((self.nclasses, -1)))[:, -1]
			else:
				return np.zeros(self.nclasses)
		else:
			if self.fit_intercept:
				return self.optimizer.x[-1]
			else:
				return 0.0

	def predict(self, X):
		"""
		Predict the class of new observations

		Parameters
		----------
		X : array(n_samples, n_features)
			Input data on which to predict classes.

		Returns
		-------
		pred : array(n_samples, )
			Predicted class for each observation
		"""
		if self._is_mult:
			return np.argmax(_pred_fun_mult(self.optimizer.x, X, self.nclasses), axis=1)
		else:
			return (_pred_fun_bin(self.optimizer.x, X) >= .5).astype('uint8')
	def predict_proba(self, X):
		"""
		Predict class probabilities for new observations

		Parameters
		----------
		X : array(n_samples, n_features)
			Input data on which to predict class probabilities.

		Returns
		-------
		pred : array(n_samples, n_classes)
			Predicted class probabilities for each observation
		"""
		if self._is_mult:
			return _pred_fun_mult(self.optimizer.x, X, self.nclasses)
		else:
			pred = _pred_fun_bin(self.optimizer.x, X).reshape((-1, 1))
			out = np.c_[1 - pred, pred]
			return out

	def _check_fit_inp(self, X, y, sample_weights):
		if sample_weights is None:
			sample_weights = np.ones(X.shape[0])
		else:
			sample_weights = sample_weights.reshape(-1)
		assert sample_weights.shape[0] == X.shape[0]
		assert X.shape[0] == y.shape[0]
		X = _StochQN._check_sp_type(self, X)
		if isspmatrix(y):
			warnings.warn("'StochasticLogisticRegression' only supports dense arrays for 'y', will cast the array.")
			y = np.array(y.todense())
		sample_weights /= X.shape[0] ### scikit-learn's function compute sums instead of means
		return X, y, sample_weights

	def _initialize_optimizer(self, X, y):
		if self.optimizer is None:
			if len(y.shape) == 1:
				self._is_mult = False
				self.nclasses = 2
				obj_fun = _obj_fun_bin
				grad_fun = _grad_fun_bin
				hess_vec_fun = _hessvec_fun_bin
				pred_fun = _pred_fun_bin
			else:
				self._is_mult = True
				self.nclasses = y.shape[1]
				obj_fun = _obj_fun_mult
				grad_fun = _grad_fun_multi
				hess_vec_fun = _hessvec_fun_mult
				pred_fun = _pred_fun_mult
			np.random.seed(self.random_state)
			w0 = np.random.normal(size = (X.shape[1] + self.fit_intercept) * (y.shape[1] if self._is_mult else 1))
			if self.optimizer_name == "oLBFGS":
				self.optimizer = oLBFGS(x0=w0, grad_fun=grad_fun, obj_fun=obj_fun, pred_fun=pred_fun, **self.optimizer_kwargs)
			elif self.optimizer_name == "SQN":
				self.optimizer = SQN(x0=w0, grad_fun=grad_fun, obj_fun=obj_fun, pred_fun=pred_fun, hess_vec_fun=hess_vec_fun, **self.optimizer_kwargs)
			elif self.optimizer_name == "adaQN":
				self.optimizer = adaQN(x0=w0, grad_fun=grad_fun, obj_fun=obj_fun, pred_fun=pred_fun, **self.optimizer_kwargs)
			else:
				raise ValueError("'optimizer' must be one of 'oLBFGS', 'SQN', or 'adaQN'.")

	def fit(self, X, y, sample_weights=None):
		"""
		Fit Logistic Regression model in stochastic batches

		Parameters
		----------
		X : array(n_samples, n_features)
			Covariates (features).
		y : array(n_samples, ) or array(n_samples, n_classes)
			Labels for each observation (must be already one-hot encoded).
		sample_weights : array(n_samples, ) or None
			Observation weights for each data point.

		Returns
		-------
		self : obj
			This object
		"""
		X, y, sample_weights = self._check_fit_inp(X, y, sample_weights)
		self._initialize_optimizer(X, y)
		self.optimizer.fit(X, y, sample_weights, {"reg_param" : self.reg_param})
		self.is_fitted = True
		return self

	def partial_fit(self, X, y, sample_weights=None):
		"""
		Fit Logistic Regression model in stochastic batches

		Note
		----
		The step size will not be decrease after running this function. In order to decrease
		the step size manually, you can set 'this.optimizer.step_size'.

		Parameters
		----------
		X : array(n_samples, n_features)
			Covariates (features).
		y : array(n_samples, ) or array(n_samples, n_classes)
			Labels for each observation (must be already one-hot encoded).
		sample_weights : array(n_samples, ) or None
			Observation weights for each data point.

		Returns
		-------
		self : obj
			This object
		"""
		X, y, sample_weights = self._check_fit_inp(X, y, sample_weights)
		self._initialize_optimizer(X, y)
		decr_step_size_before = self.optimizer.decr_step_size
		self.optimizer.decr_step_size = None
		self.optimizer.partial_fit(X, y, sample_weights, {"reg_param" : self.reg_param})
		self.optimizer.decr_step_size = decr_step_size_before
		self.is_fitted = True
		return self
