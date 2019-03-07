import numpy as np
cimport numpy as np
import ctypes, multiprocessing
from libc.string cimport memcpy
cimport cython

cdef extern from "../include/stochqn.h":
	ctypedef struct bfgs_mem:
		double *s_mem
		double *y_mem
		double *buffer_rho
		double *buffer_alpha
		double *s_bak
		double *y_bak
		size_t mem_size
		size_t mem_used
		size_t mem_st_ix
		size_t upd_freq
		double y_reg
		double min_curvature

	ctypedef struct fisher_mem:
		double *F
		double *buffer_y
		size_t mem_size
		size_t mem_used
		size_t mem_st_ix

	ctypedef struct workspace_oLBFGS:
		bfgs_mem *bfgs_memory
		double *grad_prev
		double hess_init
		size_t niter
		int section
		int nthreads
		int check_nan
		int n

	ctypedef struct workspace_SQN:
		bfgs_mem *bfgs_memory
		double *grad_prev
		double *x_sum
		double *x_avg_prev
		int use_grad_diff
		size_t niter
		int section
		int nthreads
		int check_nan
		int n

	ctypedef struct workspace_adaQN:
		bfgs_mem *bfgs_memory
		fisher_mem *fisher_memory
		double *H0
		double *grad_prev
		double *x_sum
		double *x_avg_prev
		double *grad_sum_sq
		double f_prev
		double max_incr
		double scal_reg
		double rmsprop_weight
		int use_grad_diff
		size_t niter
		int section
		int nthreads
		int check_nan
		int n

	ctypedef enum task_enum:
		calc_grad = 101
		calc_grad_same_batch = 102
		calc_grad_big_batch = 103
		calc_hess_vec = 104
		calc_fun_val_batch = 105
		invalid_input = 100

	ctypedef enum info_enum:
		func_increased = 201
		curvature_too_small = 202
		search_direction_was_nan = 203
		no_problems_encountered = 200

	int run_oLBFGS(double step_size, double *x, double *grad, double **req, task_enum *task, workspace_oLBFGS *oLBFGS, info_enum *iter_info)
	int run_SQN(double step_size, double *x, double *grad, double *hess_vec, double **req, double **req_vec, task_enum *task, workspace_SQN *SQN, info_enum *iter_info)
	int run_adaQN(double step_size, double *x, double f, double *grad, double **req, task_enum *task, workspace_adaQN *adaQN, info_enum *iter_info)

### Tasks will be outputed from python as strings rather than enums
task_dct = {
	101 : 'calc_grad',
	102 : 'calc_grad_same_batch',
	103 : 'calc_grad_big_batch',
	104 : 'calc_hess_vec',
	105 : 'calc_fun_val_batch'
}

info_dct = {
	201 : 'func_increased',
	202 : 'curvature_too_small',
	203 : 'search_direction_was_nan',
	200 : 'no_problems_encountered'
}

### The structs are redone here as Cython classes in order to make them garbage-collected
### and for making their attributes accessible from Python too
cdef class _cy_bfgs_mem:
	cdef bfgs_mem bfgs_memory
	cdef double[:] s_mem
	cdef double[:] y_mem
	cdef double[:] buffer_rho
	cdef double[:] buffer_alpha
	cdef double[:] s_bak
	cdef double[:] y_bak
	cdef public int n

	def __init__(self, int n, size_t mem_size, size_t upd_freq, double min_curvature, double y_reg):
		self.n = n

		self.s_mem = np.empty(n * mem_size, dtype=ctypes.c_double)
		self.y_mem = np.empty(n * mem_size, dtype=ctypes.c_double)
		self.buffer_rho = np.empty(mem_size, dtype=ctypes.c_double)
		self.buffer_alpha = np.empty(mem_size, dtype=ctypes.c_double)

		self.bfgs_memory.mem_size = <size_t> mem_size
		self.bfgs_memory.mem_used = <size_t> 0
		self.bfgs_memory.mem_st_ix = <size_t> 0
		self.bfgs_memory.upd_freq = <size_t> upd_freq
		self.bfgs_memory.min_curvature = min_curvature
		self.bfgs_memory.y_reg = y_reg
		self.bfgs_memory.s_mem = &self.s_mem[0]
		self.bfgs_memory.y_mem = &self.y_mem[0]
		self.bfgs_memory.buffer_rho = &self.buffer_rho[0]
		self.bfgs_memory.buffer_alpha = &self.buffer_alpha[0]

		if min_curvature > 0:
			self.s_bak = np.empty(n, dtype=ctypes.c_double)
			self.y_bak = np.empty(n, dtype=ctypes.c_double)
			self.bfgs_memory.s_bak = &self.s_bak[0]
			self.bfgs_memory.y_bak = &self.y_bak[0]

	cdef bfgs_mem* get_pointer(self):
		return &self.bfgs_memory

	@property
	def s_mem(self):
		return np.array(self.s_mem[:self.n * self.bfgs_memory.mem_used]).reshape((self.bfgs_memory.mem_used, self.n))

	@property
	def y_mem(self):
		return np.array(self.y_mem[:self.n * self.bfgs_memory.mem_used]).reshape((self.bfgs_memory.mem_used, self.n))

	@property
	def buffer_rho(self):
		return np.array(self.buffer_rho)

	@property
	def buffer_alpha(self):
		return np.array(self.buffer_alpha)

	@property
	def s_bak(self):
		if self.bfgs_memory.min_curvature > 0:
			return np.array(self.s_bak)
		else:
			return None

	@property
	def y_bak(self):
		if self.bfgs_memory.min_curvature > 0:
			return np.array(self.y_bak)
		else:
			return None

cdef class _cy_fisher_mem:
	cdef fisher_mem fisher_memory
	cdef double[:] F
	cdef double[:] buffer_y
	cdef public int n

	def __init__(self, int n, size_t mem_size):
		self.fisher_memory.mem_size = <size_t> mem_size
		self.fisher_memory.mem_used = <size_t> 0
		self.fisher_memory.mem_st_ix = <size_t> 0
		self.n = n
		self.F = np.empty(n * mem_size, dtype=ctypes.c_double)
		self.buffer_y = np.empty(n, dtype=ctypes.c_double)
		self.fisher_memory.F = &self.F[0]
		self.fisher_memory.buffer_y = &self.buffer_y[0]

	cdef fisher_mem* get_pointer(self):
		return &self.fisher_memory

	@property
	def F(self):
		size_arr = self.fisher_memory.mem_used * self.n
		return np.array(self.F[:size_arr]).reshape((self.fisher_memory.mem_used, self.n))

	@property
	def buffer_y(self):
		return np.array(self.buffer_y)

cdef class _cy_oLBFGS:
	cdef _cy_bfgs_mem _cy_bfgs_memory
	cdef double[:] grad_prev
	cdef workspace_oLBFGS oLBFGS

	def __init__(self, int n, size_t mem_size, double hess_init, double min_curvature, double y_reg, int check_nan, int nthreads):
		self._cy_bfgs_memory = _cy_bfgs_mem(n, mem_size, 1, min_curvature, y_reg)
		self.grad_prev = np.empty(n, dtype=ctypes.c_double)
		self.oLBFGS.n = n
		self.oLBFGS.hess_init = hess_init
		self.oLBFGS.niter = <size_t> 0
		self.oLBFGS.section = <int> 0
		self.oLBFGS.nthreads = nthreads
		self.oLBFGS.check_nan = check_nan
		self.oLBFGS.bfgs_memory = self._cy_bfgs_memory.get_pointer()
		self.oLBFGS.grad_prev = &self.grad_prev[0]

	cdef workspace_oLBFGS* get_pointer(self):
		return &self.oLBFGS

	@property
	def bfgs_memory(self):
		return self._cy_bfgs_memory

	@property
	def grad_prev(self):
		return np.array(self.grad_prev)

	@property
	def niter(self):
		return self.oLBFGS.niter

	@property
	def section(self):
		return self.oLBFGS.section

cdef class _cy_SQN:
	cdef _cy_bfgs_mem _cy_bfgs_memory
	cdef double[:] grad_prev
	cdef double[:] x_sum
	cdef double[:] x_avg_prev
	cdef workspace_SQN SQN

	def __init__(self, int n, size_t mem_size, size_t bfgs_upd_freq, double min_curvature,
		double y_reg, int use_grad_diff, int check_nan, int nthreads):

		self._cy_bfgs_memory = _cy_bfgs_mem(n, mem_size, bfgs_upd_freq, min_curvature, y_reg)
		if use_grad_diff:
			self.grad_prev = np.empty(n, dtype=ctypes.c_double)
			self.SQN.grad_prev = &self.grad_prev[0]
		self.x_sum = np.zeros(n, dtype=ctypes.c_double)
		self.x_avg_prev = np.empty(n, dtype=ctypes.c_double)
		self.SQN.use_grad_diff = use_grad_diff
		self.SQN.niter = <size_t> 0
		self.SQN.section = <int> 0
		self.SQN.nthreads = nthreads
		self.SQN.check_nan = check_nan
		self.SQN.n = n
		self.SQN.bfgs_memory = self._cy_bfgs_memory.get_pointer()

		self.SQN.x_sum = &self.x_sum[0]
		self.SQN.x_avg_prev = &self.x_avg_prev[0]

	cdef workspace_SQN* get_pointer(self):
		return &self.SQN

	@property
	def bfgs_memory(self):
		return self._cy_bfgs_memory

	@property
	def grad_prev(self):
		if self.use_grad_diff:
			return np.array(self.grad_prev)
		else:
			return None

	@property
	def x_avg(self):
		nused = self.SQN.niter % self._cy_bfgs_memory.bfgs_memory.upd_freq
		if nused > 0:
			return np.array(self.x_sum) / nused
		else:
			return None

	@property
	def x_avg_prev(self):
		if self.SQN.niter < self._cy_bfgs_memory.bfgs_memory.upd_freq:
			return None
		else:
			return np.array(self.x_avg_prev)

	@property
	def niter(self):
		return self.SQN.niter

	@property
	def section(self):
		return self.SQN.section

cdef class _cy_adaQN:
	cdef _cy_bfgs_mem _cy_bfgs_memory
	cdef _cy_fisher_mem _cy_fisher_memory
	cdef double[:] H0
	cdef double[:] grad_prev
	cdef double[:] x_sum
	cdef double[:] x_avg_prev
	cdef double[:] grad_sum_sq
	cdef workspace_adaQN adaQN

	def __init__(self, int n, size_t mem_size, size_t fisher_size, size_t bfgs_upd_freq, 
		double max_incr, double min_curvature, double scal_reg, double rmsprop_weight,
		double y_reg, int use_grad_diff, int check_nan, int nthreads):

		self._cy_bfgs_memory = _cy_bfgs_mem(n, mem_size, bfgs_upd_freq, min_curvature, y_reg)
		if not use_grad_diff:
			self._cy_fisher_memory = _cy_fisher_mem(n, fisher_size)
			self.adaQN.fisher_memory = self._cy_fisher_memory.get_pointer()
			self.adaQN.grad_prev = NULL
		else:
			self.grad_prev = np.empty(n, dtype=ctypes.c_double)
			self.adaQN.grad_prev = &self.grad_prev[0]
			self.adaQN.fisher_memory = NULL
		self.H0 = np.empty(n, dtype=ctypes.c_double)
		self.x_sum = np.zeros(n, dtype=ctypes.c_double)
		self.x_avg_prev = np.empty(n, dtype=ctypes.c_double)
		self.grad_sum_sq = np.zeros(n, dtype=ctypes.c_double)

		self.adaQN.f_prev = <double> 0
		self.adaQN.niter = <size_t> 0
		self.adaQN.section = <int> 0
		self.adaQN.max_incr = max_incr
		self.adaQN.scal_reg = scal_reg
		self.adaQN.use_grad_diff = use_grad_diff
		self.adaQN.nthreads = nthreads
		self.adaQN.check_nan = check_nan
		self.adaQN.n = n
		self.adaQN.rmsprop_weight = rmsprop_weight
		self.adaQN.bfgs_memory = self._cy_bfgs_memory.get_pointer()

		self.adaQN.H0 = &self.H0[0]
		self.adaQN.x_sum = &self.x_sum[0]
		self.adaQN.x_avg_prev = &self.x_avg_prev[0]
		self.adaQN.grad_sum_sq = &self.grad_sum_sq[0]

	cdef workspace_adaQN* get_pointer(self):
		return &self.adaQN

	@property
	def bfgs_memory(self):
		return self._cy_bfgs_memory

	@property
	def fisher_memory(self):
		if not self.adaQN.use_grad_diff:
			return self._cy_fisher_memory
		else:
			return None

	@property
	def H0(self):
		return np.array(self.H0)

	@property
	def x_avg(self):
		nused = self.adaQN.niter % self._cy_bfgs_memory.bfgs_memory.upd_freq
		if nused > 0:
			return np.array(self.x_sum) / nused
		else:
			return None

	@property
	def x_avg_prev(self):
		if self.adaQN.niter < self._cy_bfgs_memory.bfgs_memory.upd_freq:
			return None
		else:
			return np.array(self.x_avg_prev)

	@property
	def grad_sum_sq(self):
		return np.array(self.grad_sum_sq)

	@property
	def grad_prev(self):
		if self.adaQN.use_grad_diff:
			return np.array(self.grad_prev)
		else:
			return None

	@property
	def niter(self):
		return self.adaQN.niter

	@property
	def section(self):
		return self.adaQN.section

cdef double* _take_np_ptr(np.ndarray[double, ndim=1] a):
	return &a[0]

cdef class _StochQN_free:
	cdef double *req_ptr
	cdef double[:] gradient
	cdef public info_enum iter_info
	cdef public task_enum task

	cdef public size_t mem_size
	cdef public double min_curvature
	cdef public double y_reg
	cdef public int check_nan
	cdef public int nthreads
	cdef public int initialized

	def _take_common_inputs(self, mem_size, min_curvature, y_reg, check_nan, nthreads):
		assert mem_size > 0
		assert isinstance(mem_size, int)

		if min_curvature is not None:
			assert min_curvature > 0
		else:
			min_curvature = 0
		if y_reg is not None:
			assert y_reg > 0
		else:
			y_reg = 0

		if nthreads == -1:
			nthreads = multiprocessing.cpu_count()
		if nthreads is None:
			nthreads = 1
		assert isinstance(nthreads, int)
		assert nthreads >= 1
		self.mem_size = <size_t> mem_size
		self.min_curvature = <double> min_curvature
		self.y_reg = <double> y_reg
		self.check_nan = <int> bool(check_nan)
		self.nthreads = <int> nthreads

	@property
	def gradient(self):
		return np.array(self.gradient)

	def update_gradient(self, gradient):
		"""
		Pass requested gradient to optimizer

		Parameters
		----------
		gradient : array(m, )
			Gradient calculated as requested, evaluated at values given in "requested_on",
			calcualted either in a regular batch (task = "calc_grad"), same batch as before
			(task = "calc_grad_same_batch" - oLBFGS only), or a larger batch of data (task = "calc_grad_big_batch"), perhaps
			including all the cases from the last such calculation (SQN and adaQN with 'use_grad_diff=True').
		"""
		self.gradient = gradient.astype('float64').reshape(-1)

@cython.embedsignature(True)
cdef class oLBFGS_free(_StochQN_free):
	"""
	oLBFGS optimizer (free mode)

	Optimizes an empirical (convex) loss function over batches of sample data. Compared to
	class 'oLBFGS', this version lets the user do all the calculations from the outside, only
	interacting with the object by mode of a function that returns a request type and is fed the
	required calculation through a method 'update_gradient'.

	Order in which requests are made:

	========== loop ===========
	* calc_grad
	* calc_grad_same_batch		(might skip if using check_nan)
	===========================

	Parameters
	----------
	mem_size : int
		Number of correction pairs to store for approximation of Hessian-vector products.
	hess_init : float or None
		value to which to initialize the diagonal of H0.
		If passing 0, will use the same initializion as for SQN (s_last*y_last / y_last*y_last).
	y_reg : float or None
		regularizer for 'y' vector (gets added y_reg * s)
	min_curvature : float or None
		Minimum value of s*y / s*s in order to accept a correction pair.
	check_nan : bool
		Whether to check for variables becoming NaN after each iteration, and reverting the step if they do
		(will also reset BFGS memory).
	nthreads : int
		Number of parallel threads to use. If set to -1, will determine the number of available threads and use
		all of them. Note however that not all the computations can be parallelized.
	"""

	cdef public _cy_oLBFGS _oLBFGS
	cdef public double hess_init

	def __init__(self, mem_size=10, hess_init=None, min_curvature=1e-4, y_reg=None, check_nan=True, nthreads=-1):
		self._take_common_inputs(mem_size, min_curvature, y_reg, check_nan, nthreads)
		if hess_init is not None:
			assert hess_init > 0
		else:
			hess_init = 0

		self.hess_init = <double> hess_init
		self.initialized = 0

	def _initialize(self, n):
		self._oLBFGS = _cy_oLBFGS(<int> n, self.mem_size, self.hess_init,
			self.min_curvature, self.y_reg, self.check_nan, self.nthreads)
		self.gradient = np.empty(n, dtype=ctypes.c_double)
		self.initialized = 1

	def run_optimizer(self, x, step_size):
		"""
		Continue optimization process after supplying the calculation requested from the last run

		Continue the optimization process from where it was left since the last calculation was
		requested. Will internally do all the updates that are possible until the moment some
		calculation of function/gradient/hessian-vector is required.

		Note
		----
		The first time this is run, no calculation needs to be supplied.

		Parameters
		----------
		x : array(m, )
			Current values of the variables. Will be modified in-place.
			Do NOT modify the values between runs.
		step_size : float
			Step size for the next update (note that variables are not updated during all runs).

		Returns
		-------
		request : dict
			Dictionary with the calculation required to proceed and iteration information.
			Structure:
				* task : str - one of "calc_grad", "calc_grad_same_batch" (oLBFGS w. 'min_curvature' or 'check_nan'),
				"calc_hess_vec" (SQN wo. 'use_grad_diff'), "calc_fun_val_batch" (adaQN w. 'max_incr'),
				"calc_grad_big_batch" (SQN and adaQN w. 'use_grad_diff').
				* requested_on : array(m, ) or tuple(array(m, ), array(m, )), containing the values on which
				the request in "task" has to be evaluated. In the case of Hessian-vector products (SQN), the
				first vector is the values of 'x' and the second is the vector with which the product is required.
				* info : dict(x_changed_in_run : bool, iteration_number : int, iteration_info : str),
				iteration_info can be one of "no_problems_encountered", "search_direction_was_nan",
				"func_increased", "curvature_too_small".
		"""
		assert isinstance(x, np.ndarray)
		assert x.dtype == np.float64
		if not self.initialized:
			self._initialize(x.shape[0])

		x_changed = run_oLBFGS(<double> step_size, _take_np_ptr(x), &self.gradient[0], &self.req_ptr, &self.task, self._oLBFGS.get_pointer(), &self.iter_info)

		if self.req_ptr == _take_np_ptr(x):
			req_arr = x
		else:
			req_arr = np.asarray(<np.float64_t[:self._oLBFGS.oLBFGS.n]> self.req_ptr)

		out = {
			"task" : task_dct[int(self.task)],
			"requested_on" : req_arr,
			"info" : {
				"x_changed_in_run" : bool(x_changed),
				"iteration_number" : <int> self._oLBFGS.oLBFGS.niter,
				"iteration_info" : info_dct[int(self.iter_info)]
			}
		}
		return out

@cython.embedsignature(True)
cdef class SQN_free(_StochQN_free):
	"""
	SQN optimizer (free mode)

	Optimizes an empirical (convex) loss function over batches of sample data. Compared to
	class 'SQN', this version lets the user do all the calculations from the outside, only
	interacting with the object by mode of a function that returns a request type and is fed the
	required calculation through methods 'update_gradient' and 'update_hess_vec'.

	Order in which requests are made:

	========== loop ===========
	* calc_grad
		... (repeat calc_grad)
	if 'use_grad_diff':
		* calc_grad_big_batch
	else:
		* calc_hess_vec
	===========================

	Parameters
	----------
	mem_size : int
		Number of correction pairs to store for approximation of Hessian-vector products.
	bfgs_upd_freq : int
		Number of iterations (batches) after which to generate a BFGS correction pair.
	min_curvature : float or None
		Minimum value of s*y / s*s in order to accept a correction pair.
	use_grad_diff : bool
		Whether to create the correction pairs using differences between gradients instead of Hessian-vector products.
		These gradients are calculated on a larger batch than the regular ones (given by batch_size * bfgs_upd_freq).
	check_nan : bool
		Whether to check for variables becoming NaN after each iteration, and reverting the step if they do
		(will also reset BFGS memory).
	nthreads : int
		Number of parallel threads to use. If set to -1, will determine the number of available threads and use
		all of them. Note however that not all the computations can be parallelized.
	"""

	cdef double *req_vec_ptr
	cdef double[:] hess_vec
	cdef public _cy_SQN _SQN

	cdef public size_t bfgs_upd_freq
	cdef public int use_grad_diff

	def __init__(self, mem_size=10, bfgs_upd_freq=20, min_curvature=1e-4, y_reg=None, use_grad_diff=False, check_nan=True, nthreads=-1):
		self._take_common_inputs(mem_size, min_curvature, y_reg, check_nan, nthreads)
		assert bfgs_upd_freq > 0

		self.bfgs_upd_freq = <size_t> int(bfgs_upd_freq)
		self.use_grad_diff = <int> bool(use_grad_diff)

		self.initialized = 0

	def _initialize(self, n):
		self._SQN = _cy_SQN(<int> n, self.mem_size, self.bfgs_upd_freq, self.min_curvature,
			self.y_reg, self.use_grad_diff, self. check_nan, self.nthreads)
		self.gradient = np.empty(n, dtype=ctypes.c_double)
		if not self.use_grad_diff:
			self.hess_vec = np.empty(n, dtype=ctypes.c_double)
		else:
			self.hess_vec = np.empty(1, dtype=ctypes.c_double)
		self.initialized = 1

	def update_hess_vec(self, hess_vec):
		"""
		Pass requested Hessian-vector product to optimizer (task = "calc_hess_vec")
		
		Parameters
		----------
		hess_vec : array(m, )
			Product of the Hessian evaluated at "requested_on"[0] with the vector "requested_on"[1],
			calculated a larger batch of data than the gradient, perhaps including all the cases from the last such calculation.
		"""
		self.hess_vec = hess_vec.astype('float64').reshape(-1)

	def run_optimizer(self, x, step_size):
		"""
		Continue optimization process after supplying the calculation requested from the last run

		Continue the optimization process from where it was left since the last calculation was
		requested. Will internally do all the updates that are possible until the moment some
		calculation of function/gradient/hessian-vector is required.

		Note
		----
		The first time this is run, no calculation needs to be supplied.

		Parameters
		----------
		x : array(m, )
			Current values of the variables. Will be modified in-place.
		step_size : float
			Step size for the next update (note that variables are not updated during all runs).

		Returns
		-------
		request : dict
			Dictionary with the calculation required to proceed and iteration information.
			Structure:
				* task : str - one of "calc_grad", "calc_grad_same_batch" (oLBFGS w. 'min_curvature' or 'check_nan'),
				"calc_hess_vec" (SQN wo. 'use_grad_diff'), "calc_fun_val_batch" (adaQN w. 'max_incr'),
				"calc_grad_big_batch" (SQN and adaQN w. 'use_grad_diff').
				* requested_on : array(m, ) or tuple(array(m, ), array(m, )), containing the values on which
				the request in "task" has to be evaluated. In the case of Hessian-vector products (SQN), the
				first vector is the values of 'x' and the second is the vector with which the product is required.
				* info : dict(x_changed_in_run : bool, iteration_number : int, iteration_info : str),
				iteration_info can be one of "no_problems_encountered", "search_direction_was_nan",
				"func_increased", "curvature_too_small".
		"""
		assert isinstance(x, np.ndarray)
		if not self.initialized:
			self._initialize(x.shape[0])

		x_changed = run_SQN(<double> step_size, _take_np_ptr(x), &self.gradient[0], &self.hess_vec[0],
			&self.req_ptr, &self.req_vec_ptr, &self.task, self._SQN.get_pointer(), &self.iter_info)

		if self.req_ptr == _take_np_ptr(x):
			req_arr = x
		elif self.req_ptr == &self._SQN.x_sum[0]:
			req_arr = np.array(self._SQN.x_sum)
		else:
			req_arr = np.asarray(<np.float64_t[:self._SQN.SQN.n]> self.req_ptr)

		
		out = {
			"task" : task_dct[int(self.task)],
			"requested_on" : None,
			"info" : {
				"x_changed_in_run" : bool(x_changed),
				"iteration_number" : <int> self._SQN.SQN.niter,
				"iteration_info" : info_dct[int(self.iter_info)]
			}
		}
		if task_dct[int(self.task)] == "calc_hess_vec":
			out["requested_on"] = (req_arr, np.asarray(<np.float64_t[:self._SQN.SQN.n]> self.req_vec_ptr))
		else:
			out["requested_on"] = req_arr


		return out

	@property
	def hess_vec(self):
		if self.use_grad_diff:
			return None
		else:
			return np.array(self.hess_vec)

@cython.embedsignature(True)
cdef class adaQN_free(_StochQN_free):
	"""
	adaQN optimizer (free mode)

	Optimizes an empirical (perhaps non-convex) loss function over batches of sample data. Compared to
	class 'adaQN', this version lets the user do all the calculations from the outside, only
	interacting with the object by mode of a function that returns a request type and is fed the
	required calculation through methods 'update_gradient' and 'update_function'.

	Order in which requests are made:

	========== loop ===========
	* calc_grad
		... (repeat calc_grad)
	if max_incr > 0:
		* calc_fun_val_batch
	if 'use_grad_diff':
		* calc_grad_big_batch	(skipped if below max_incr)
	===========================

	Parameters
	----------
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
	scal_reg : float
		Regularization parameter to use in the denominator for AdaGrad and RMSProp scaling.
	rmsprop_weight : float(0,1) or None
		If not 'None', will use RMSProp formula instead of AdaGrad for approximated inverse-Hessian initialization.
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
	"""
	
	cdef public double f
	cdef public _cy_adaQN _adaQN
	
	cdef public size_t fisher_size
	cdef public size_t bfgs_upd_freq
	cdef public double max_incr
	cdef public double scal_reg
	cdef public double rmsprop_weight
	cdef public int use_grad_diff

	def __init__(self, mem_size=10, fisher_size=100, bfgs_upd_freq=20, max_incr=1.01, min_curvature=1e-4, scal_reg=1e-4,
		rmsprop_weight=None, y_reg=None, use_grad_diff=False, check_nan=True, nthreads=-1):

		self._take_common_inputs(mem_size, min_curvature, y_reg, check_nan, nthreads)
		assert bfgs_upd_freq > 0
		bfgs_upd_freq = int(bfgs_upd_freq)
		if not use_grad_diff:
			assert fisher_size > 0
			fisher_size = int(fisher_size)
		else:
			fisher_size = 0
		if max_incr is not None:
			assert max_incr > 0
		else:
			max_incr = 0
		assert scal_reg > 0
		if rmsprop_weight is not None:
			assert rmsprop_weight > 0
			assert rmsprop_weight < 1
		else:
			rmsprop_weight = 0

		self.fisher_size = <size_t> fisher_size
		self.bfgs_upd_freq = <size_t> bfgs_upd_freq
		self.max_incr = <double> max_incr
		self.scal_reg = <double> scal_reg
		self.rmsprop_weight = <double> rmsprop_weight
		self.use_grad_diff = <int> bool(use_grad_diff)

		self.initialized = 0

	def _initialize(self, n):
		self._adaQN = _cy_adaQN(<int> n, self.mem_size, self.fisher_size, self.bfgs_upd_freq, self.max_incr,
			self.min_curvature, self.scal_reg, self.rmsprop_weight, self.y_reg, self.use_grad_diff, self.check_nan, self.nthreads)
		self.gradient = np.empty(n, dtype=ctypes.c_double)
		self.initialized = 1

	def update_function(self, fun):
		"""
		Pass requested function evaluation to optimizer (task = "calc_fun_val_batch")

		Parameters
		----------
		fun : float
			Function evaluated at "requested_on" under a validation set or a larger batch, perhaps
			including all the cases from the last such calculation.
		"""
		self.f = <double> fun

	def run_optimizer(self, x, step_size):
		"""
		Continue optimization process after supplying the calculation requested from the last run

		Continue the optimization process from where it was left since the last calculation was
		requested. Will internally do all the updates that are possible until the moment some
		calculation of function/gradient/hessian-vector is required.

		Note
		----
		The first time this is run, no calculation needs to be supplied.

		Parameters
		----------
		x : array(m, )
			Current values of the variables. Will be modified in-place.
			Do NOT modify the values between runs.
		step_size : float
			Step size for the next update (note that variables are not updated during all runs).

		Returns
		-------
		request : dict
			Dictionary with the calculation required to proceed and iteration information.
			Structure:
				* task : str - one of "calc_grad", "calc_grad_same_batch" (oLBFGS w. 'min_curvature' or 'check_nan'),
				"calc_hess_vec" (SQN wo. 'use_grad_diff'), "calc_fun_val_batch" (adaQN w. 'max_incr'),
				"calc_grad_big_batch" (SQN and adaQN w. 'use_grad_diff').
				* requested_on : array(m, ) or tuple(array(m, ), array(m, )), containing the values on which
				the request in "task" has to be evaluated. In the case of Hessian-vector products (SQN), the
				first vector is the values of 'x' and the second is the vector with which the product is required.
				* info : dict(x_changed_in_run : bool, iteration_number : int, iteration_info : str),
				iteration_info can be one of "no_problems_encountered", "search_direction_was_nan",
				"func_increased", "curvature_too_small".
		"""
		assert isinstance(x, np.ndarray)
		if not self.initialized:
			self._initialize(x.shape[0])

		x_changed = run_adaQN(<double> step_size, _take_np_ptr(x), self.f, &self.gradient[0], &self.req_ptr,
			&self.task, self._adaQN.get_pointer(), &self.iter_info)

		if self.req_ptr == _take_np_ptr(x):
			req_arr = x
		elif self.req_ptr == &self._adaQN.x_sum[0]:
			req_arr = np.array(self._adaQN.x_sum)
		else:
			req_arr = np.asarray(<np.float64_t[:self._adaQN.adaQN.n]> self.req_ptr)

		out = {
			"task" : task_dct[int(self.task)],
			"requested_on" : req_arr,
			"info" : {
				"x_changed_in_run" : bool(x_changed),
				"iteration_number" : <int> self._adaQN.adaQN.niter,
				"iteration_info" : info_dct[int(self.iter_info)]
			}
		}
		return out
