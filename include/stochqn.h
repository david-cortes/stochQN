/*	Stochastic limited-memory Quasi-Newton optimization

	Methods for smooth stochastic optimization of both convex and
	non-convex functions, using search directions computed by
	an approximated inverse Hessian-vector product, which is obtained
	through limited-memory BFGS recursive formula.

	The implementations are based on the following works:

	*	Byrd, R.H., Hansen, S.L., Nocedal, J. and Singer, Y., 2016.
		"A stochastic quasi-Newton method for large-scale optimization."
		SIAM Journal on Optimization, 26(2), pp.1008-1031.
		(SQN)
	* 	Schraudolph, N.N., Yu, J. and Günter, S., 2007, March.
		"A stochastic quasi-Newton method for online convex optimization."
		In Artificial Intelligence and Statistics (pp. 436-443).
		(oLBFGS)
	* 	Keskar, N.S. and Berahas, A.S., 2016, September.
		"adaQN: An Adaptive Quasi-Newton Algorithm for Training RNNs."
		In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 1-16). Springer, Cham.
		(adaQN)
	*	Wright, S. and Nocedal, J., 1999.
		"Numerical optimization." (ch 7)
		Springer Science, 35(67-68), p.7.
		(L-BFGS two-loop recursion, and correction pairs based on gradient differences)

	Written for C99 standard with fixes for compilation under MSVC.
	
	BSD 2-Clause License

	Copyright (c) 2019, David Cortes
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:

	* Redistributions of source code must retain the above copyright notice, this
	  list of conditions and the following disclaimer.

	* Redistributions in binary form must reproduce the above copyright notice,
	  this list of conditions and the following disclaimer in the documentation
	  and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
	OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


/* ====== Note: Go straight towards the end to find the function prototypes ====== */

#ifndef STOCHQN_INCLUDE
#define STOCHQN_INCLUDE 

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/*	Containers and functions for initializing each optimizer as if it were a class */
typedef struct {
	double *s_mem;
	double *y_mem;
	double *buffer_rho;
	double *buffer_alpha;
	double *s_bak;
	double *y_bak;
	size_t mem_size;
	size_t mem_used;
	size_t mem_st_ix;
	size_t upd_freq; /* L */
	double y_reg; /* lambda  in oLBFGS*/
	double min_curvature; /* adaQN: epsilon in main loop */
} bfgs_mem;

typedef struct {
	double *F;
	double *buffer_y;
	size_t mem_size;
	size_t mem_used;
	size_t mem_st_ix;
} fisher_mem;

typedef struct {
	bfgs_mem *bfgs_memory;
	double *grad_prev;
	double hess_init; /* epsilon */
	size_t niter;
	int section; /* do NOT modify!!! */
	int nthreads;
	int check_nan;
	int n;
} workspace_oLBFGS;

typedef struct {
	bfgs_mem *bfgs_memory;
	double *grad_prev;
	double *x_sum; /* w_bar */
	double *x_avg_prev;
	int use_grad_diff;
	size_t niter;
	int section; /* do NOT modify!!! */
	int nthreads;
	int check_nan;
	int n;
} workspace_SQN;

typedef struct {
	bfgs_mem *bfgs_memory;
	fisher_mem *fisher_memory;
	double *H0;
	double *grad_prev;
	double *x_sum; /* w_bar */
	double *x_avg_prev;
	double *grad_sum_sq;
	double f_prev; /* can modify if the validation batch is changed */
	double max_incr; /* gamma */
	double scal_reg; /* epsilon (in gradient rescaling) */
	double rmsprop_weight;
	int use_grad_diff;
	size_t niter;
	int section; /* do NOT modify!!! */
	int nthreads;
	int check_nan;
	int n;
} workspace_adaQN;


/*	Allocate and deallocate structs for each optimizer
	
	These structs work pretty much like a C++ class. They need
	to be allocated and deallocated by the user as needed. They are then
	passed as pointers to each respective optimizer.

	The "recommended values" refer to the values that the authors recommended in
	each respective paper, and not necessarily to what works better in practice.

	IMPORTANT: most of these values should NOT be modified manually after initialization.
		* The following can be modified at any point: 'y_reg', 'scal_reg', 'min_curvature', 'max_incr',
													  'hess_init, 'nthreads', 'check_nan', 'rmsprop_weight'
		* The following can be modified at the moment of a correction pair creation: 'upd_freq' (in 'bfgs_memory')
		* The following MUST be modified whenever the validation batch changes: 'f_prev' (adaQN with 'max_incr')


	Parameters
	==========
	n
		number of variables in the minimization problem
		(this is restricted to int type due to BLAS functions types)
	
	mem_size
		number of correction pairs to store for approximation of Hessian-vector products
		(recommended value: 10)
	
	fisher_size (adaQN)
		number of past gradients to store for approximating Fisher matrix (ignored with 'use_grad_diff')
		(recommended value: 100)
	
	bfgs_upd_freq (SQN, adaQN)
		how often to create a correction pair
		(recommended value: 10)
	
	min_curvature
		minimum value of <s,y>/<s,s> to accept a correction pair
		(recommended value: 1e-4 for adaQN, 0 for oLBFGS, no recommendation for SQN)
	
	max_incr (adaQN)
		maximum relative change in function value to accept an update (in the variables)
		(recommended value: 1.01)
	
	hess_init (oLBFGS) 
		value to which to initialize the diagonal of H0.
		If passing 0, will use the same initializion as for SQN (<s_last, y_last> / <y_last, y_last>)
		(recommended value: 1e-10)
	
	y_reg
		regularizer for 'y' vector (gets added y_reg * s)
		(recommended value: 0)
	
	scal_reg (adaQN)
		regularization term for AdaGrad and RMSProp gradient rescaling
		(recommended value: 1e-4)
	
	rmsprop_weight (adaQN)
		weight in interval(0,1) to give to past gradients on RMSProp (if 0, will use AdaGrad)
		(recommended value: 0, recommended value when non-zero: 0.9)
	
	use_grad_diff (SQN, adaQN)
		whether to calculate 'y' correction vectors as gradient differences
		(ignores Fisher matrix in adaQN, and does not request Hessian-Vector product in SQN)
		(recommended value: 0)
	
	check_nan
		(boolean) check whether the search direction as any infinite or NaN values - this is more likely to
		happen when using oLBFGS or SQN with non-convex functions, which they weren't meant for
		(if direction contains NaNs or Inf, will not take the step and will reset BFGS and Fisher memory)
		recommended value: 0)
	
	nthreads
		number of parallel threads to use (when advantageous to parallelize)
	*/
workspace_oLBFGS* initialize_oLBFGS(const int n, const size_t mem_size, const double hess_init, const double y_reg,
	const double min_curvature, const int check_nan, const int nthreads);
void dealloc_oLBFGS(workspace_oLBFGS *oLBFGS);

workspace_SQN* initialize_SQN(const int n, const size_t mem_size, const size_t bfgs_upd_freq, const double min_curvature,
	const int use_grad_diff, const double y_reg, const int check_nan, const int nthreads);
void dealloc_SQN(workspace_SQN *SQN);

workspace_adaQN* initialize_adaQN(const int n, const size_t mem_size, const size_t fisher_size, const size_t bfgs_upd_freq,
	const double max_incr, const double min_curvature, const double scal_reg, const double rmsprop_weight,
	const int use_grad_diff, const double y_reg, const int check_nan, const int nthreads);
void dealloc_adaQN(workspace_adaQN *adaQN);


/*	Indicator for next calculation required by the optimizer

	Optimizers are run in this way: they are given a workspace, pointer to an array (**req), pointer to a 'task' variable,
	step size, gradient and variables pointers, and perhaps other parameters. They modify the 'task' variable,
	which indicates which calculation is required next by the optimizer, and this calculation should be performed on
	the variable values given in *req (e.g. evaluate the gradient on the variable values given in *req).

	Once the required task (calculation) is completed, the optimizer function is run again with the
	same workspace, and updated gradient, hessian-vector, or function pointers as requested.

	Whenever the variables are updated by the optimizer (and they aren't updated at each run), this reflects
	in an increase in workspace->niter.
	
	Meanings of codes
	=================
	calc_grad												: calculate the gradient of the function on a new batch of data
	calc_grad_same_batch (oLBFGS)							: calculate the gradient of the function with the same batch of data as
															  the previous calculation
	calc_grad_big_batch (SQN & adaQN w. 'use_grad_diff')	: calculate the gradient of the function on a larger batch of data,
															  ideally using all the batches that were used from the last such calculation
	calc_hess_vec (SQN)										: calculate the product of the hessian with a vector,
															  using all the batches that were used from the last such calculation
	calc_fun_val_batch (adaQN w. max_incr>0)				: calculate the function (objective) value on a validation batch,
															  or a large batch like for calc_hess_vec
	invalid_input											: the inputs or workspace passed to the function was/were invalid
															  (optimization won't continue afterwards)
*/
typedef enum task_enum {
	calc_grad = 101,
	calc_grad_same_batch = 102,
	calc_grad_big_batch = 103,
	calc_hess_vec = 104,
	calc_fun_val_batch = 105,
	invalid_input = 100
} task_enum;


/*	Status indicators for progress during an iteration */
typedef enum info_enum {
	func_increased = 201,
	curvature_too_small = 202,
	search_direction_was_nan = 203,
	no_problems_encountered = 200
} info_enum;

/*	Whenever the weights are updated, the return value from an optimizer will reflect it.
	You can also check the iterationu number in workspace->niter.
	Note that the return values are actually of type 'int' and not 'iter_status',
	this is just a visual reminder.
*/
typedef enum iter_status {did_not_update_x = 0, updated_x = 1, received_invalid_input = -1000} iter_status;

/*	Optimization functions

	These functions work like a C++ class method - they are given a workspace (see documentation above),
	and at each call, they will make a request (see documentation above), which must be calculated externally.
	The calculation should be performed taking the values on the array pointed at by *req (e.g. 'eval_grad(*req)'),
	which is not always the same as the 'x' variables, and then the optimization should should be run again
	with everything the same except for the values of the calculation that was request.

	A C++ version using classes with RAII principles is also provided. The parameters are the same as
	for the C version.

	Step size should be set by the user for each iteration (it's not modified internally). adaQN requires
	larger step sizes than the other methods.

	The inputs, once calculated, should be passed in one of:
		- grad[] 		(gradient evaluations)
		- f 			(function evaluations)
		- hess_vec[] 	(Hessian-vector products)

	Order in which requests are made:

	oLBFGS:
		========== loop ===========
		* calc_grad
		* calc_grad_same_batch		(might skip if using check_nan)
		===========================

	SQN:
		========== loop ===========
		* calc_grad
			... (repeat calc_grad)
		if 'use_grad_diff':
			* calc_grad_big_batch
		else:
			* calc_hess_vec
		===========================

	adaQN:
		========== loop ===========
		* calc_grad
			... (repeat calc_grad)
		if max_incr > 0:
			* calc_fun_val_batch
		if 'use_grad_diff':
			* calc_grad_big_batch	(skipped if below max_incr)
		===========================

	Return values are integers corresponding to codes in 'iter_status'

	NOTE: be aware that the 'grad' array is modified in-place.

	Parameters
	==========
	step_size (in) : double
		size of the step to take in the computed direction
	
	x (in, out) : double[m]
		variables to optimize
	
	f (adaQN w. 'max_incr')	(in) : double
		objective function value, evaluated at the requested values
		(ignored for oLBFGS, SQN, and adaQN wo. 'max_incr')
	
	grad (in) : double[m]
		pointer to array with gradient evalueted at '*req' (not at 'x')
		(warning: will be modified in-place!!!)
	
	hess_vec (SQN wo. 'use_grad_diff') (in) : double[m]
		pointer to array with the product of the Hessian and the array in '*req'
		ignored for oLBFGS, adaQN, and SQN w. 'use_grad_diff')
	
	*req (out) : double[m]
		values of the variables at which the next calculation is requested
		(do NOT modify the values in this array!!!)
	
	*req_vec (out) (SQN wo. 'use_grad_diff') : double[m]
		vector with which the Hessian (evaluated at *req) should be multiplied
		(do NOT modify the values in this array!!!)
	
	task (out) : task_enum
		calculation that should be performed next
	
	workspace (in, out) : struct
		struct with the required data and allocated variables
	
	iter_info (out) : info_enum
		slot where to put information when something goes wrong (e.g. curvature too small)
*/
int run_oLBFGS(double step_size, double x[], double grad[], double **req, task_enum *task, workspace_oLBFGS *oLBFGS, info_enum *iter_info);
int run_SQN(double step_size, double x[], double grad[], double hess_vec[], double **req, double **req_vec, task_enum *task, workspace_SQN *SQN, info_enum *iter_info);
int run_adaQN(double step_size, double x[], double f, double grad[], double **req, task_enum *task, workspace_adaQN *adaQN, info_enum *iter_info);


#ifdef __cplusplus
}
#endif


/*	C++ 'safe' objects - these follow RAII principles
	
	API is the same as for the C version, but variables and pointers are stored in a class.
	See the documentation above for more details, or the (straightforward) code for what
	they do.
*/
#ifdef __cplusplus
#include <new>

class oLBFGS
{
public:
	workspace_oLBFGS *workspace;
	task_enum task;
	info_enum info;
	iter_status status;
	double *req;

	oLBFGS(const int n, const size_t mem_size = 10, const double hess_init = 0, const double y_reg = 0,
		   const double min_curvature = 0, const int check_nan = 1, const int nthreads = 1)
	{
		this->workspace = initialize_oLBFGS(n, mem_size, hess_init, y_reg,
											min_curvature, check_nan, nthreads);
		if (this->workspace == NULL) throw std::bad_alloc();
		this->task   = calc_grad;
		this->status = did_not_update_x;
		this->info   = no_problems_encountered;
	}
	~oLBFGS() { if (this->workspace != NULL) dealloc_oLBFGS(this->workspace); }


	iter_status run(double step_size, double x[], double grad[])
	{
		return (iter_status) run_oLBFGS(step_size, x, grad, &this->req, &this->task,
										this->workspace, &this->info);
	}
	task_enum get_task()      { return this->task;             }
	info_enum get_iter_info() { return this->info;             }
	size_t    get_n_iter()    { return this->workspace->niter; }
	double*   get_req()       { return this->req;              }
};


class SQN
{
public:
	workspace_SQN *workspace;
	task_enum task;
	info_enum info;
	iter_status status;
	double *req;
	double *req_vec;

	SQN(const int n, const size_t mem_size = 10, const size_t bfgs_upd_freq = 10,
		const double min_curvature = 1e-4, const int use_grad_diff = 0, const double y_reg = 0,
		const int check_nan = 1, const int nthreads = 1)
	{
		this->workspace = initialize_SQN(n, mem_size, bfgs_upd_freq, min_curvature, use_grad_diff,
										 y_reg, check_nan, nthreads);
		if (this->workspace == NULL) throw std::bad_alloc();
		this->task   = calc_grad;
		this->status = did_not_update_x;
		this->info   = no_problems_encountered;
	}

	~SQN() { if (this->workspace != NULL) dealloc_SQN(this->workspace); }

	iter_status run(double step_size, double x[], double grad[], double hess_vec[])
	{
		return (iter_status) run_SQN(step_size, x, grad, hess_vec, &this->req, &this->req_vec,
									 &this->task, this->workspace, &this->info);
	}


	task_enum get_task()      { return this->task;             }
	info_enum get_iter_info() { return this->info;             }
	size_t    get_n_iter()    { return this->workspace->niter; }
	double*   get_req()       { return this->req;              }
	double*   get_req_vec()   { return this->req_vec;          }

};

class adaQN
{
public:
	workspace_adaQN *workspace;
	task_enum task;
	info_enum info;
	iter_status status;
	double *req;

	adaQN(const int n, const size_t mem_size = 10, const size_t fisher_size = 100,
		  const size_t bfgs_upd_freq = 10, const double max_incr = 1.01, const double min_curvature = 1e-4,
		  const double scal_reg = 1e-4, const double rmsprop_weight = 0.9, const int use_grad_diff = 0,
		  const double y_reg = 0, const int check_nan = 1, const int nthreads = 1)
	{
		this->workspace = initialize_adaQN(n, mem_size, fisher_size, bfgs_upd_freq,
										   max_incr, min_curvature, scal_reg, rmsprop_weight,
										   use_grad_diff, y_reg, check_nan, nthreads);
		if (this->workspace == NULL) throw std::bad_alloc();
		this->task   = calc_grad;
		this->status = did_not_update_x;
		this->info   = no_problems_encountered;
	}

	~adaQN() { if (this->workspace != NULL) dealloc_adaQN(this->workspace); }

	iter_status run(double step_size, double x[], double f, double grad[])
	{
		return (iter_status) run_adaQN(step_size, x, f, grad, &this->req,
									   &this->task, this->workspace, &this->info);
	}

	task_enum get_task()      { return this->task;             }
	info_enum get_iter_info() { return this->info;             }
	size_t    get_n_iter()    { return this->workspace->niter; }
	double*   get_req()       { return this->req;              }
};


#endif /* __cplusplus */


#endif /* STOCHQN_INCLUDE */
