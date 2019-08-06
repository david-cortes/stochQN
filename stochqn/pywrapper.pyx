import numpy as np
cimport numpy as np
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

cdef double* ptr_double(np.ndarray[double, ndim = 1] a):
	return &a[0]

cdef bfgs_mem get_c_BFGS_mem(py_BFGS_mem):
	cdef bfgs_mem BFGS_mem
	BFGS_mem.s_mem = ptr_double(py_BFGS_mem.s_mem)
	BFGS_mem.y_mem = ptr_double(py_BFGS_mem.y_mem)
	BFGS_mem.buffer_rho = ptr_double(py_BFGS_mem.buffer_rho)
	BFGS_mem.buffer_alpha = ptr_double(py_BFGS_mem.buffer_alpha)
	BFGS_mem.s_bak = ptr_double(py_BFGS_mem.s_bak)
	BFGS_mem.y_bak = ptr_double(py_BFGS_mem.y_bak)
	BFGS_mem.mem_size = py_BFGS_mem.mem_size
	BFGS_mem.mem_used = py_BFGS_mem.mem_used
	BFGS_mem.mem_st_ix = py_BFGS_mem.mem_st_ix
	BFGS_mem.upd_freq = py_BFGS_mem.upd_freq
	BFGS_mem.y_reg = py_BFGS_mem.y_reg
	BFGS_mem.min_curvature = py_BFGS_mem.min_curvature
	return BFGS_mem

cdef fisher_mem get_c_Fisher_mem(py_Fisher_mem):
	cdef fisher_mem Fisher_mem
	Fisher_mem.F = ptr_double(py_Fisher_mem.F)
	Fisher_mem.buffer_y = ptr_double(py_Fisher_mem.buffer_y)
	Fisher_mem.mem_size = py_Fisher_mem.mem_size
	Fisher_mem.mem_used = py_Fisher_mem.mem_used
	Fisher_mem.mem_st_ix = py_Fisher_mem.mem_st_ix
	return Fisher_mem

cdef workspace_oLBFGS get_c_oLBFGS(py_oLBFGS, bfgs_mem *BFGS_mem):
	cdef workspace_oLBFGS oLBFGS
	oLBFGS.bfgs_memory = BFGS_mem
	oLBFGS.grad_prev = ptr_double(py_oLBFGS.grad_prev)
	oLBFGS.hess_init = py_oLBFGS.hess_init
	oLBFGS.niter = py_oLBFGS.niter
	oLBFGS.section = py_oLBFGS.section
	oLBFGS.nthreads = py_oLBFGS.nthreads
	oLBFGS.check_nan = py_oLBFGS.check_nan
	oLBFGS.n = py_oLBFGS.n
	return oLBFGS

cdef workspace_SQN get_c_SQN(py_SQN, bfgs_mem *BFGS_mem):
	cdef workspace_SQN SQN
	SQN.bfgs_memory = BFGS_mem
	SQN.grad_prev = ptr_double(py_SQN.grad_prev)
	SQN.x_sum = ptr_double(py_SQN.x_sum)
	SQN.x_avg_prev = ptr_double(py_SQN.x_avg_prev)
	SQN.use_grad_diff = py_SQN.use_grad_diff
	SQN.niter = py_SQN.niter
	SQN.section = py_SQN.section
	SQN.nthreads = py_SQN.nthreads
	SQN.check_nan = py_SQN.check_nan
	SQN.n = py_SQN.n
	return SQN

cdef workspace_adaQN get_c_adaQN(py_adaQN, bfgs_mem *BFGS_mem, fisher_mem *Fisher_mem):
	cdef workspace_adaQN adaQN
	adaQN.bfgs_memory = BFGS_mem
	adaQN.fisher_memory = Fisher_mem
	adaQN.H0 = ptr_double(py_adaQN.H0)
	adaQN.grad_prev = ptr_double(py_adaQN.grad_prev)
	adaQN.x_sum = ptr_double(py_adaQN.x_sum)
	adaQN.x_avg_prev = ptr_double(py_adaQN.x_avg_prev)
	adaQN.grad_sum_sq = ptr_double(py_adaQN.grad_sum_sq)
	adaQN.f_prev = py_adaQN.f_prev
	adaQN.max_incr = py_adaQN.max_incr
	adaQN.scal_reg = py_adaQN.scal_reg
	adaQN.rmsprop_weight = py_adaQN.rmsprop_weight
	adaQN.use_grad_diff = py_adaQN.use_grad_diff
	adaQN.niter = py_adaQN.niter
	adaQN.section = py_adaQN.section
	adaQN.nthreads = py_adaQN.nthreads
	adaQN.check_nan = py_adaQN.check_nan
	adaQN.n = py_adaQN.n
	return adaQN

def _py_run_oLBFGS(py_oLBFGS, x, grad, step_size):
	cdef bfgs_mem BFGS_mem = get_c_BFGS_mem(py_oLBFGS.BFGS_mem)
	cdef workspace_oLBFGS oLBFGS = get_c_oLBFGS(py_oLBFGS, &BFGS_mem)
	cdef info_enum iter_info
	cdef task_enum task
	cdef double* req
	cdef int x_changed = run_oLBFGS(step_size, ptr_double(x), ptr_double(grad), &req, &task, &oLBFGS, &iter_info)
	return  x_changed, oLBFGS.niter, oLBFGS.section, \
			BFGS_mem.mem_used, BFGS_mem.mem_st_ix, \
			task, iter_info, np.asarray(<np.float64_t[:x.shape[0]]> req)

def py_run_SQN(py_SQN, x, step_size, grad, hess_vec):
	cdef bfgs_mem BFGS_mem = get_c_BFGS_mem(py_SQN.BFGS_mem)
	cdef workspace_SQN SQN = get_c_SQN(py_SQN, &BFGS_mem)
	cdef info_enum iter_info
	cdef task_enum task
	cdef double* req
	cdef double* req_vec
	cdef int x_changed = run_SQN(step_size, ptr_double(x), ptr_double(grad), ptr_double(hess_vec),
								 &req, &req_vec, &task, &SQN, &iter_info)
	if task == calc_hess_vec:
		return  x_changed, SQN.niter, SQN.section, \
				SQN.bfgs_memory.mem_used, SQN.bfgs_memory.mem_st_ix, \
				task, iter_info, np.asarray(<np.float64_t[:x.shape[0]]> req), np.asarray(<np.float64_t[:x.shape[0]]> req_vec)
	else:
		return  x_changed, SQN.niter, SQN.section, \
				BFGS_mem.mem_used, BFGS_mem.mem_st_ix, \
				task, iter_info, np.asarray(<np.float64_t[:x.shape[0]]> req), None

def py_run_adaQN(py_adaQN, x, grad, step_size, f):
	cdef bfgs_mem BFGS_mem = get_c_BFGS_mem(py_adaQN.BFGS_mem)
	cdef fisher_mem Fisher_mem = get_c_Fisher_mem(py_adaQN.Fisher_mem)
	cdef workspace_adaQN adaQN = get_c_adaQN(py_adaQN, &BFGS_mem, &Fisher_mem)
	cdef info_enum iter_info
	cdef task_enum task
	cdef double* req = NULL
	cdef int x_changed = run_adaQN(step_size, ptr_double(x), f, ptr_double(grad), &req, &task, &adaQN, &iter_info)
	return  x_changed, adaQN.niter, adaQN.section, \
			BFGS_mem.mem_used, BFGS_mem.mem_st_ix, \
			Fisher_mem.mem_used, Fisher_mem.mem_st_ix, adaQN.f_prev, \
			task, iter_info, np.asarray(<np.float64_t[:x.shape[0]]> req)
