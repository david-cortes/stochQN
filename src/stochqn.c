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

	Written for C99 standard with fixes for compilation with OpenMP 2.0 (e.g. MSVC).
	
	BSD 2-Clause License

	Copyright (c) 2020, David Cortes
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

/* Standard headers */
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#ifdef _OPENMP
	#include <omp.h>
#endif
#ifndef _FOR_R
	#include <stdio.h>
#endif

/* Library header */
#include "stochqn.h"

/* BLAS functions */
#ifdef _FOR_PYTON
	#include "findblas.h" /* https://www.github.com/david-cortes/findblas */
#elif defined(_FOR_R)
	#include "blas_R.h"
	#include <R_ext/Print.h>
	#define fprintf(f, message) REprintf(message)
#else
	#include "blasfuns.h"
#endif


/*	--------------- Preprocessor definitions ---------------	*/

/* Aliasing for compiler optimizations */
#ifdef __cplusplus
	#if defined(__GNUG__) || defined(__GNUC__) || defined(_MSC_VER) || defined(__clang__) || defined(__INTEL_COMPILER)
		#define restrict __restrict
	#else
		#define restrict 
	#endif
#elif defined(_MSC_VER)
	#define restrict __restrict
#elif !defined(__STDC_VERSION__) || (__STDC_VERSION__ < 199901L)
	#define restrict 
#endif

/* In-lining for faster calls */
#ifndef __cplusplus
	#if defined(_MSC_VER)
		#define inline __inline
	#elif !defined(__STDC_VERSION__) || (__STDC_VERSION__ < 199901L)
		#define inline 
	#endif
#endif

/*	OpenMP < 3.0 (e.g. MSVC as of 2019) does not support parallel for's with unsigned iterators,
	and does not support declaring the iterator type in the loop itself */
#ifdef _OPENMP
	#if (_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64) /* OpenMP < 3.0 */
		#define size_t_for 
	#else
		#define size_t_for size_t
	#endif
#else
	#define size_t_for size_t
#endif

#ifndef isnan
	#ifdef _isnan
		#define isnan _isnan
	#else
		#define isnan(x) ( (x) != (x) )
	#endif
#endif
#ifndef isinf
	#ifdef _finite
		#define isinf(x) (!_finite(x))
	#else
		#define isinf(x) ( (x) >= HUGE_VAL || (x) <= -HUGE_VAL )
	#endif
#endif

#define x_avg x_sum /* this is to keep track of when the sum array has been divided */

#define min2(a, b) (((a) < (b))? (a) : (b))

/*	--------------- End of preprocessor definitions ---------------	*/


#ifdef __cplusplus
extern "C" {
#endif
/*	--------------- General-purpose helpers ---------------	*/
static inline void copy_arr(const real_t *restrict src, real_t *restrict dest, const int n, const int nthreads)
{
	/* Note: don't use BLAS dcopy as it's actually much slower */
	#if defined(_OPENMP)

	int i;
	int chunk_size = n / nthreads;
	int remainder = n % nthreads;

	/* oracle compilers cannot take 'const int' (CRAN requirement for building in solaris OS) */
	int nthreads_non_const = min2(nthreads, 2);
	/* Note: on x86, using more than 2 threads will end up making it slower */

	#pragma omp parallel for schedule(static, 1) firstprivate(src, dest, chunk_size, nthreads) num_threads(nthreads_non_const)
	for (i = 0; i < nthreads; i++){
		memcpy(dest + i * chunk_size, src + i * chunk_size, sizeof(real_t) * chunk_size);
	}
	if (remainder > 0){
		memcpy(dest + nthreads * chunk_size, src + nthreads * chunk_size, sizeof(real_t) * remainder);
	}

	#else
	memcpy(dest, src, sizeof(real_t) * n);
	#endif
}

static inline void set_to_zero(real_t arr[], const int n, const int nthreads)
{

	#if defined(_OPENMP)

	int i;
	int chunk_size = n / nthreads;
	int remainder = n % nthreads;
	/* oracle compilers cannot take 'const int' (CRAN requirement for building in solaris OS) */
	int nthreads_non_const = min2(nthreads, 2);
	/* Note: on x86 CPUs, using more than 2 threads will make it slower */

	#pragma omp parallel for schedule(static, 1) firstprivate(arr, chunk_size, nthreads) num_threads(nthreads_non_const)
	for (i = 0; i < nthreads; i++){
		memset(arr + i * chunk_size, 0, sizeof(real_t) * chunk_size);
	}
	if (remainder > 0){
		memset(arr + nthreads * chunk_size, 0, sizeof(real_t) * remainder);
	}

	#else
	memset(arr, 0, sizeof(real_t) * n);
	#endif
}

static inline void multiply_elemwise(real_t *restrict inout, const real_t *restrict other, const int n, const int nthreads)
{
	#if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64)) /* OpenMP < 3.0 */
	int i;
	int n_szt = n;
	#else
	size_t n_szt = (size_t) n;
	#endif

	/* oracle compilers cannot take 'const int' (CRAN requirement for building in solaris OS) */
	int nthreads_non_const = nthreads;

	#pragma omp parallel for if((n > 1e7) && (nthreads > 4)) schedule(static) firstprivate(inout, other, n_szt) num_threads(nthreads_non_const)
	for (size_t_for i = 0; i < n_szt; i++) inout[i] *= other[i];
}

static inline void difference_elemwise(real_t *restrict out, const real_t *restrict later, const real_t *restrict earlier, const int n, const int nthreads)
{
	#if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64)) /* OpenMP < 3.0 */
	int i;
	int n_szt = n;
	#else
	size_t n_szt = (size_t) n;
	#endif

	/* oracle compilers cannot take 'const int' (CRAN requirement for building in solaris OS) */
	int nthreads_non_const = nthreads;
	
	#pragma omp parallel for if( (n > 1e7) && (nthreads > 4)) schedule(static) firstprivate(n_szt, out, later, earlier) num_threads(nthreads_non_const)
	for (size_t_for i = 0; i < n_szt; i++) out[i] = later[i] - earlier[i];
}

static inline int check_inf_nan(const real_t arr[], const int n, const int nthreads)
{

	size_t n_szt = (size_t) n;
	int is_wrong = 0;

	#if defined(_OPENMP) & !defined(_WIN32) &!defined(_WIN64) & (_OPENMP > 201305) /* OpenMP >= 4.0 */
	/*	Note1: in most cases the array should not have invalid elements
		Note2: 'omp cancel' is disabled by default through an environmental variable,
				and it will ignore modifications of it within the same calling program,
				so it very likely willnot end up cancelling for most use-cases.
	*/

	/* oracle compilers cannot take 'const int' (CRAN requirement for building in solaris OS) */
	int nthreads_non_const = nthreads;
	if ( (n > 1e8) && (nthreads > 4) ){
		#pragma omp parallel for schedule(static) firstprivate(arr, n_szt) reduction(max: is_wrong) num_threads(nthreads_non_const)
		for (size_t i = 0; i < n_szt; i++){
			if (isinf(arr[i])){
				is_wrong = 1;
				// #pragma omp cancel for
			}

			if (isnan(arr[i])){
				is_wrong = 1;
				// #pragma omp cancel for
			}
		}
	} else
	#endif
	{
		for (size_t i = 0; i < n_szt; i++){
			if (isinf(arr[i])){return 1;}
			if (isnan(arr[i])){return 1;}
		}
	}
	if (is_wrong){return 1;}
	return 0;
}

static inline void add_to_sum(const real_t *restrict new_values, real_t *restrict sum_arr, const size_t n, const int nthreads)
{
	/* Note: daxpy in MKL is actually slower than this */

	#if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64)) /* OpenMP < 3.0 */
	int i;
	int n_szt = n;
	#else
	size_t n_szt = (size_t) n;
	#endif

	/* oracle compilers cannot take 'const int' (CRAN requirement for building in solaris OS) */
	int nthreads_non_const = nthreads;

	#pragma omp parallel for if((n > 1e7) && (nthreads_non_const > 4)) schedule(static) firstprivate(sum_arr, new_values, n_szt) num_threads(nthreads_non_const)
	for (size_t_for i = 0; i < n_szt; i++) sum_arr[i] += new_values[i];
}

static inline void average_from_sum(real_t arr_sum[], const size_t n_summed, const int n)
{
	if (n_summed > 1){
		cblas_tscal(n, 1 / (real_t) n_summed, arr_sum, 1);
	}
}
/*	--------------- End of general-purpose helpers ---------------	*/


/*	Optimizers have a workspace that works pretty much like a C++ class.
	This is a long piece of code dealing with memory management,
	you'll probably want to skip it.	*/

/*	--------- Beginning of initializers, deallocators, and updaters --------	*/
bfgs_mem* initialize_bfgs_mem(const size_t mem_size, const int n, const real_t min_curvature, const real_t y_reg, const size_t upd_freq)
{
	real_t *s_bak;
	real_t *y_bak;
	if (min_curvature > 0){
		s_bak = (real_t*) malloc(sizeof(real_t) * n);
		y_bak = (real_t*) malloc(sizeof(real_t) * n);
	} else {
		s_bak = NULL;
		y_bak = NULL;
	}
	real_t *s_mem = (real_t*) malloc(sizeof(real_t) * n * mem_size);
	real_t *y_mem = (real_t*) malloc(sizeof(real_t) * n * mem_size);
	real_t *buffer_rho = (real_t*) malloc(sizeof(real_t) * mem_size);
	real_t *buffer_alpha = (real_t*) malloc(sizeof(real_t) * mem_size);
	bfgs_mem *out = (bfgs_mem*) malloc(sizeof(bfgs_mem));
	out->s_mem = s_mem;
	out->y_mem = y_mem;
	out->buffer_rho = buffer_rho;
	out->buffer_alpha = buffer_alpha;
	out->s_bak = s_bak;
	out->y_bak = y_bak;
	out->mem_size = mem_size;
	out->mem_used = 0;
	out->mem_st_ix = 0;
	out->upd_freq = upd_freq;
	out->y_reg = y_reg;
	out->min_curvature = min_curvature;
	return out;
}

void dealloc_bfgs_mem(bfgs_mem *bfgs_memory)
{
	free(bfgs_memory->s_mem);
	free(bfgs_memory->y_mem);
	free(bfgs_memory->buffer_rho);
	free(bfgs_memory->buffer_alpha);
	free(bfgs_memory->s_bak);
	free(bfgs_memory->y_bak);
	free(bfgs_memory);
}

fisher_mem* initialize_fisher_mem(const size_t mem_size, const int n)
{
	real_t *F = (real_t*) malloc(sizeof(real_t) * n * mem_size);
	real_t *buffer_y = (real_t*) malloc(sizeof(real_t) * mem_size);
	fisher_mem *out = (fisher_mem*) malloc(sizeof(fisher_mem));
	out->F = F;
	out->buffer_y = buffer_y;
	out->mem_size = mem_size;
	out->mem_used = 0;
	out->mem_st_ix = 0;
	return out;
}

void dealloc_fisher_mem(fisher_mem *fisher_memory)
{
	free(fisher_memory->F);
	free(fisher_memory->buffer_y);
	free(fisher_memory);
}

static inline int check_bfgsmem_nonnull(bfgs_mem* bfgs_memory)
{
	if (
		(bfgs_memory->y_mem == NULL) ||
		(bfgs_memory->s_mem == NULL) ||
		(bfgs_memory->buffer_rho == NULL) ||
		(bfgs_memory->buffer_alpha == NULL) ||
		(bfgs_memory->s_bak == NULL && bfgs_memory->min_curvature > 0) ||
		(bfgs_memory->y_bak == NULL && bfgs_memory->min_curvature > 0)
		) 	{
			fprintf(stderr, "Error: Could not allocate memory for BFGS storage.\n");
			return 1;
			}
	return 0;
}

static inline int check_fishermem_nonnull(fisher_mem* fisher_memory)
{
	if (fisher_memory->F == NULL || fisher_memory->buffer_y == NULL){
		fprintf(stderr, "Error: Could not allocate memory for Fisher storage.\n");
		return 1;
	}
	return 0;
}

static inline int check_oLBFGS_nonnull(workspace_oLBFGS *oLBFGS)
{
	/* Check for memory allocation failure */
	if ( (oLBFGS->bfgs_memory == NULL) ||
		(oLBFGS->grad_prev == NULL) ||
		(oLBFGS == NULL) ){
			fprintf(stderr, "Error: Could not allocate memory for oLBFGS.\n");
			return 1;
	}
	return check_bfgsmem_nonnull(oLBFGS->bfgs_memory);
}

static inline int check_SQN_nonnull(workspace_SQN *SQN)
{
	/* Check for memory allocation failure */
	if ( (SQN->bfgs_memory == NULL) ||
		(SQN->x_sum == NULL) ||
		(SQN->x_avg_prev == NULL) ||
		(SQN->grad_prev == NULL && SQN->use_grad_diff) ||
		(SQN == NULL) ){
			dealloc_SQN(SQN);
			fprintf(stderr, "Error: Could not allocate memory for SQN.\n");
			return 1;
	}
	return check_bfgsmem_nonnull(SQN->bfgs_memory);
}

static inline int check_adaQN_nonnull(workspace_adaQN *adaQN)
{
	/* Check for memory allocation failure */
	if ( (adaQN->bfgs_memory == NULL) ||
		(adaQN->H0 == NULL) ||
		(adaQN->x_sum == NULL) ||
		(adaQN->x_avg_prev == NULL) ||
		(adaQN->grad_sum_sq == NULL) ||
		(adaQN->grad_prev == NULL && adaQN->use_grad_diff) ||
		(adaQN == NULL) ){
			dealloc_adaQN(adaQN);
			fprintf(stderr, "Error: Could not allocate memory for adaQN.\n");
			return 1;
	}
	if (  check_bfgsmem_nonnull(adaQN->bfgs_memory)  ) {return 1;};
	if (!adaQN->use_grad_diff){return check_fishermem_nonnull(adaQN->fisher_memory);}

	return 0;
}

void dealloc_oLBFGS(workspace_oLBFGS *oLBFGS)
{
	dealloc_bfgs_mem(oLBFGS->bfgs_memory);
	free(oLBFGS->grad_prev);
	free(oLBFGS);
}

void dealloc_SQN(workspace_SQN *SQN)
{
	dealloc_bfgs_mem(SQN->bfgs_memory);
	free(SQN->grad_prev);
	free(SQN->x_sum);
	free(SQN->x_avg_prev);
	free(SQN);
}

void dealloc_adaQN(workspace_adaQN *adaQN)
{
	dealloc_bfgs_mem(adaQN->bfgs_memory);
	if (!adaQN->use_grad_diff || adaQN->fisher_memory != NULL){
		dealloc_fisher_mem(adaQN->fisher_memory);
	}
	free(adaQN->H0);
	free(adaQN->grad_prev);
	free(adaQN->x_sum);
	free(adaQN->x_avg_prev);
	free(adaQN->grad_sum_sq);
	free(adaQN);
}

workspace_oLBFGS* initialize_oLBFGS(const int n, const size_t mem_size, const real_t hess_init,
	const real_t y_reg, const real_t min_curvature, const int check_nan, const int nthreads)
{
	bfgs_mem *bfgs_memory = initialize_bfgs_mem(mem_size, n, min_curvature, y_reg, 1);
	real_t *grad_prev = (real_t*) malloc(sizeof(real_t) * n);

	workspace_oLBFGS *out = (workspace_oLBFGS*) malloc(sizeof(workspace_oLBFGS));
	out->bfgs_memory = bfgs_memory;
	out->grad_prev = grad_prev;
	out->hess_init = hess_init;
	out->niter = 0;
	out->section = 0;
	out->check_nan = check_nan;
	out->nthreads = nthreads;
	out->n = n;
	if ( check_oLBFGS_nonnull(out) ) {dealloc_oLBFGS(out); return NULL;}
	return out;
}

workspace_SQN* initialize_SQN(const int n, const size_t mem_size, const size_t bfgs_upd_freq, const real_t min_curvature,
	const int use_grad_diff, const real_t y_reg, const int check_nan, const int nthreads)
{
	real_t *grad_prev;
	if (use_grad_diff){grad_prev = (real_t*) malloc(sizeof(real_t) * n);}
	else {grad_prev = NULL;}
	bfgs_mem *bfgs_memory = initialize_bfgs_mem(mem_size, n, min_curvature, y_reg, bfgs_upd_freq);
	real_t *x_sum = (real_t*) calloc(n, sizeof(real_t));
	real_t *x_avg_prev = (real_t*) malloc(sizeof(real_t) * n);

	workspace_SQN* out = (workspace_SQN*) malloc(sizeof(workspace_SQN));
	out->bfgs_memory = bfgs_memory;
	out->grad_prev = grad_prev;
	out->x_sum = x_sum;
	out->x_avg_prev = x_avg_prev;
	out->use_grad_diff = use_grad_diff;
	out->niter = 0;
	out->section = 0;
	out->check_nan = check_nan;
	out->nthreads = nthreads;
	out->n = n;
	if ( check_SQN_nonnull(out) ) {dealloc_SQN(out); return NULL;}
	return out;
}

workspace_adaQN* initialize_adaQN(const int n, const size_t mem_size, const size_t fisher_size, const size_t bfgs_upd_freq,
	const real_t max_incr, const real_t min_curvature, const real_t scal_reg, const real_t rmsprop_weight,
	const int use_grad_diff, const real_t y_reg, const int check_nan, const int nthreads)
{
	bfgs_mem *bfgs_memory = initialize_bfgs_mem(mem_size, n, min_curvature, y_reg, bfgs_upd_freq);
	fisher_mem *fisher_memory;
	real_t *grad_prev;
	if (use_grad_diff){
		fisher_memory = NULL;
		grad_prev = (real_t*) malloc(sizeof(real_t) * n);
	} else {
		fisher_memory = initialize_fisher_mem(fisher_size, n);
		grad_prev = NULL;
	}
	real_t *H0 = (real_t*) malloc(sizeof(real_t) * n);
	real_t *x_sum = (real_t*) calloc(n, sizeof(real_t));
	real_t *x_avg_prev = (real_t*) malloc(sizeof(real_t) * n);
	real_t *grad_sum_sq = (real_t*) calloc(n, sizeof(real_t));

	workspace_adaQN *out = (workspace_adaQN*) malloc(sizeof(workspace_adaQN));
	out->bfgs_memory = bfgs_memory;
	out->fisher_memory = fisher_memory;
	out->H0 = H0;
	out->grad_prev = grad_prev;
	out->x_sum = x_sum;
	out->x_avg_prev = x_avg_prev;
	out->grad_sum_sq = grad_sum_sq;
	out->max_incr = max_incr;
	out->scal_reg = scal_reg;
	out->rmsprop_weight = rmsprop_weight;
	out->use_grad_diff = use_grad_diff;
	out->f_prev = 0;
	out->niter = 0;
	out->section = 0;
	out->check_nan = check_nan;
	out->nthreads = nthreads;
	out->n = n;
	if ( check_adaQN_nonnull(out) ){dealloc_adaQN(out); return NULL;}
	return out;
}

/*	Functions for adding and discarding correction pairs and previous gradients.

	When deleted, the data is not overwritten or freed, but the indexes are reset to
	act as if they were not present.
*/
static inline void flush_bfgs_mem(bfgs_mem *bfgs_memory)
{
	bfgs_memory->mem_used = 0;
	bfgs_memory->mem_st_ix = 0;
}

static inline void flush_fisher_mem(fisher_mem *fisher_memory)
{
	if (fisher_memory != NULL)
	{
		fisher_memory->mem_used = 0;
		fisher_memory->mem_st_ix = 0;
	}
}

static inline void incr_bfgs_counters(bfgs_mem *bfgs_memory)
{
	bfgs_memory->mem_st_ix = (bfgs_memory->mem_st_ix + 1) % bfgs_memory->mem_size;
	bfgs_memory->mem_used = ((bfgs_memory->mem_used + 1) >= bfgs_memory->mem_size)? bfgs_memory->mem_size : (bfgs_memory->mem_used + 1);
}

static inline void incr_fisher_counters(fisher_mem *fisher_memory)
{
	fisher_memory->mem_st_ix = (fisher_memory->mem_st_ix + 1) % fisher_memory->mem_size;
	fisher_memory->mem_used = ((fisher_memory->mem_used + 1) >= fisher_memory->mem_size)? fisher_memory->mem_size : (fisher_memory->mem_used + 1);
}

static inline void add_to_fisher_mem(real_t grad[], fisher_mem *fisher_memory, const int n, const int nthreads)
{
	if (fisher_memory != NULL){
		copy_arr(grad, fisher_memory->F + fisher_memory->mem_st_ix * n, n, nthreads);
		incr_fisher_counters(fisher_memory);
	}
}

static inline void backup_corr_pair(bfgs_mem *bfgs_memory, const int n, const int nthreads)
{
	if (bfgs_memory->min_curvature > 0){
		copy_arr(bfgs_memory->s_bak, bfgs_memory->s_mem + bfgs_memory->mem_st_ix * n, n, nthreads);
		copy_arr(bfgs_memory->y_bak, bfgs_memory->y_mem + bfgs_memory->mem_st_ix * n, n, nthreads);
	}
}

static inline void rollback_corr_pair(bfgs_mem *bfgs_memory, const int n, info_enum *iter_info, const int nthreads)
{
	if (bfgs_memory->min_curvature > 0){
		copy_arr(bfgs_memory->s_bak, bfgs_memory->s_mem + bfgs_memory->mem_st_ix * n, n, nthreads);
		copy_arr(bfgs_memory->y_bak, bfgs_memory->y_mem + bfgs_memory->mem_st_ix * n, n, nthreads);
		*iter_info = curvature_too_small;
	}	
}

static inline void archive_x_avg(real_t x_avg[], real_t x_avg_prev[], const int n, const int nthreads)
{
	copy_arr(x_avg, x_avg_prev, n, nthreads);
	set_to_zero(x_sum, n, nthreads); /* x_avg is aliased to x_sum */
}
/*	--------- End of initializers, deallocators, and updaters --------	*/

/*	============= Optimization algorithms section =============
	
	Note: the functions here oftentimes have an input variable 'nthreads',
	but most of the work is done through BLAS functions, and the number
	of threads for them is set beforehand in the optimizer functions.
*/


/*	Approximate H^(-1) * g through the "L-BFGS two-loop recursion"

	For the variable names, refer to:
	Wright, S. and Nocedal, J., 1999. "Numerical optimization." (ch. 7)
	
	grad (in, out)	: real_t[n]
		Gradient for the current values of the variables - the computed search
		direction will be written to this same array, overwriting the gradient.
	n : int
		Number of variables (dimensionality of 'x')
	H0 : real_t[n] or NULL
		Initial matrix H0 (diagonal only) from which H^-1 is updated.
		If passing NULL here and zero to 'h0', will use a scalar value as suggested in the book
		"Numerical optimization." (Wright & Nocedal)
	h0 : real_t
		number to which to initialize the diagonal H0.
		If passing zero here and NULL to 'H0', will use a scalar value as suggested in the book
		"Numerical optimization." (Wright & Nocedal)
	y_mem : real_t[mem_size, n]
		'y' correction variables.
		These shall be ordered from earliest to latest, with the earliest vector
		not necessarily at the first position.
	s_mem : real_t[mem_size, n]
		's' correction variables.
		These shall be ordered from earliest to latest, with the earliest vector
		not necessarily at the first position.
	mem_size : size_t
		Dimensionality of the arrays 'y_mem' and 's_mem' (how many rows it can have).
	mem_used : size_t
		Number of filled rows in 'y_mem' and 's_mem'
	mem_st_ix : size_t
		Position in 'y_mem' and 's_mem' at which the earliest vector is stored, with later elements
		following onwards, continuing at the beginning after position 'mem_used' if this is not zero.
	buffer_rho : real_t[mem_size]
		Temporary array in which to store the computed rho values.
	buffer_alpha : real_t[mem_size]
		Temporary array in which to store the computed alpha values.
	nthreads : int
		Number of parallel threads to use - most of the work is done on a BLAS library
		(and the threads for it are set elsewhere), but for very large problems, passes
		over the grad/out array can also be parallelized.
*/
static inline void approx_inv_hess_grad(real_t grad[], int n, real_t H0[], real_t h0,
	real_t y_mem[], real_t s_mem[], size_t mem_size, size_t mem_used, size_t mem_st_ix,
	real_t buffer_rho[], real_t buffer_alpha[], int nthreads)
{
	real_t scaling, beta;
	size_t i, ipos, last_pos;

	/* backward pass: alpha <- rho * s' q; q <- q - alpha * y */
	for (size_t ii = 0; ii < mem_used; ii++)
	{
		i = mem_used - ii - 1;
		ipos = (mem_st_ix + i) % mem_size;

		buffer_rho[i] = 1 / cblas_tdot(n, y_mem + ipos*n, 1, s_mem + ipos*n, 1);
		buffer_alpha[i] = buffer_rho[i] * cblas_tdot(n, grad, 1, s_mem + ipos*n, 1);
		cblas_taxpy(n, -buffer_alpha[i], y_mem + ipos*n, 1, grad, 1);
	}

	/*	Use a diagonal matrix as a starting point:
		By default, will calculate it from the last correction pair */
	if ( (H0 == NULL) && (h0 <= 0) )
	{
		last_pos = (mem_st_ix - 1 + mem_used) % mem_size;
		scaling = cblas_tdot(n, s_mem + last_pos*n, 1, y_mem + last_pos*n, 1)
				/ cblas_tdot(n, y_mem + last_pos*n, 1, y_mem + last_pos*n, 1);
		cblas_tscal(n, scaling, grad, 1);
	}

	/*	But can also initialize it from values supplied by the user */
	else 
	{
		/* Use diagonal passed by user */
		if (H0 != NULL) { multiply_elemwise(grad, H0, n, nthreads); }

		/* Use scalar passed by user */
		else { cblas_tscal(n, h0, grad, 1); }
	}

	/* forward pass: beta <- rho * y' * r; r <- r * s * (alpha - beta) */
	for (size_t i = 0; i < mem_used; i++)
	{
		ipos = (mem_st_ix + i) % mem_size;
		beta = buffer_rho[i] * cblas_tdot(n, y_mem + ipos*n, 1, grad, 1);
		cblas_taxpy(n, buffer_alpha[i] - beta, s_mem + ipos*n, 1, grad, 1);
	}
}

/*	Update the data on previous squared gradients
	
	Can use either AdaGrad (simple sum) or RMSProp (squared sum)

	grad					: new gradient to add
	grad_sum_sq (in, out)	: array where to store sum of squared past gradients
	rmsprop_weight			: weight in interval(0,1) to give to old info (if 0, will use AdaGrad)
	n						: number of variables (dimensionality of 'x')
	nthreads				: number of parallel threads to use
*/
static inline void update_sum_sq(real_t *restrict grad, real_t *restrict grad_sum_sq, real_t rmsprop_weight, int n, int nthreads)
{
	#if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
	int n_szt = n;
	int i;
	#else
	size_t n_szt = (size_t) n;
	#endif
	real_t weight_new;

	/* oracle compilers cannot take 'const int' (CRAN requirement for building in solaris OS) */
	int nthreads_non_const = nthreads;

	/* RMSProp update */
	if (rmsprop_weight > 0 && rmsprop_weight < 1)
	{
		weight_new = 1 - rmsprop_weight;
		#pragma omp parallel for if( (n > 1e7) && (nthreads_non_const > 4)) schedule(static) firstprivate(n_szt, grad, grad_sum_sq, rmsprop_weight, weight_new) num_threads(nthreads_non_const)
		for (size_t_for i = 0; i < n_szt; i++) grad_sum_sq[i] = rmsprop_weight*grad_sum_sq[i] + weight_new*(grad[i] * grad[i]);
	}
	
	/* AdaGrad update */
	else 
	{
		#pragma omp parallel for if( (n > 1e7) && (nthreads_non_const > 4)) schedule(static) firstprivate(n_szt, grad, grad_sum_sq) num_threads(nthreads_non_const)
		for (size_t_for i = 0; i < n_szt; i++) grad_sum_sq[i] += grad[i] * grad[i];
	}
}

/*	Compute a search direction (used as H0 initializer by adaQN) as rescaled gradient using a
	diagonal matrix, given by the sums of squares of past gradients (AdaGrad or RMSProp formulae).

	direction (out)			: array where to save the computed direction.
							  (if NULL, will save the direction in the same 'grad' array)
	grad (in, out)			: current gradient
	grad_sum_sq	(in, out)	: sum of squares of past gradients (weighted sum for RMSProp)
	n						: number of variables (dimensionality of 'x')
	scal_reg				: regularization (epsilon) for the scaling
	rmsprop_weight			: weight for old gradients if using RMSProp
							  (pass 0 for AdaGrad init)
	num_threads				: number of parallel threads to use
*/
static inline void diag_rescal(real_t *restrict direction, real_t *restrict grad, real_t *restrict grad_sum_sq,
	int n, real_t scal_reg, real_t rmsprop_weight, int nthreads)
{
	update_sum_sq(grad, grad_sum_sq, rmsprop_weight, n, nthreads);
	#if defined(_OPENMP) && ((_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64))
	int i;
	int n_szt = n;
	#else
	size_t n_szt = (size_t) n;
	#endif

	/* oracle compilers cannot take 'const int' (CRAN requirement for building in solaris OS) */
	int nthreads_non_const = nthreads;

	if (direction == NULL) {
		#pragma omp parallel for if( (n > 1e7) && (nthreads_non_const >= 4) ) schedule(static) firstprivate(direction, grad_sum_sq, scal_reg, n_szt) num_threads(nthreads_non_const)
		for (size_t_for i = 0; i < n_szt; i++) grad[i] /= sqrt(grad_sum_sq[i] + scal_reg);
	} else {
		#pragma omp parallel for if( (n > 1e7) && (nthreads_non_const >= 4) ) schedule(static) firstprivate(direction, grad_sum_sq, scal_reg, n_szt) num_threads(nthreads_non_const)
		for (size_t_for i = 0; i < n_szt; i++) direction[i] = grad[i] / sqrt(grad_sum_sq[i] + scal_reg);
	}
}

/*	Take a step in the search direction specified by the respective algorithm
	
	step_size					: size of the step to take
	n							: number of variables (dimensionality of 'x')
	x (in, out)					: current values of the variables
	grad (in, out)				: gradient at current values of x - the search direction will be
								  written there, overwriting the gradient
	bfgs_memory					: BFGS memory struct
	rmsprop_weight (adaQN)		: weight for old gradients if using RMSProp
								  (pass 0 for SQN, oLBFGS, and adaQN with AdaGrad init)
	H0 (adaQN)					: temporary array where to store diagonal initializer for inv. Hessian
	grad_sum_sq (adaQN)(in,out)	: sums of squares of past gradients (weighted sums in RMSProp)
	scal_reg (adaQN)			: regularization for the diagonal rescaling using grad_sum_sq
	check_nan					: whether to check the search direction for NaN or Inf (will reject it if so)
	iter_info					: pointer to the indicator on encountered problems
	nthreads					: number of parallel threads to use
*/
static inline void take_step(real_t step_size, int n, real_t x[], real_t grad[], bfgs_mem *bfgs_memory,
	real_t rmsprop_weight, real_t H0[], real_t h0, real_t grad_sum_sq[], real_t scal_reg,
	int check_nan, info_enum *iter_info, int nthreads)
{

	/* When there are no correction pairs, take a gradient or rescaled gradient step */
	if (bfgs_memory->mem_used == 0)
	{
		/* If no rescaling, take a simple gradient step, otherwise, take AdaGrad or RMSProp step */
		if (grad_sum_sq != NULL) {diag_rescal(NULL, grad, grad_sum_sq, n, scal_reg, rmsprop_weight, nthreads);}
	} 

	/* When there are correction pairs, get an approx. invHess-grad direction (with diagonal init) */
	else 
	{

		if (grad_sum_sq != NULL) {  diag_rescal(H0, grad, grad_sum_sq, n, scal_reg, rmsprop_weight, nthreads);  }
		approx_inv_hess_grad(grad, n, H0, h0, bfgs_memory->y_mem, bfgs_memory->s_mem,
			bfgs_memory->mem_size, bfgs_memory->mem_used, (bfgs_memory->mem_st_ix == bfgs_memory->mem_used)? 0 : bfgs_memory->mem_st_ix,
			bfgs_memory->buffer_rho, bfgs_memory->buffer_alpha, nthreads);
	}

	/* Check if the search direction is invalid */
	if (check_nan)
	{
		if ( check_inf_nan(grad, n, nthreads) ||
			/* There are also cases in which the search direction is not NaN, but is too large nevertheless */
			cblas_tnrm2(n, grad, 1) > 1e3 * n )
		{
			flush_bfgs_mem(bfgs_memory);
			*iter_info = search_direction_was_nan;
			return;
		}
	}
	
	/* Finally, take step in computed direction */
	cblas_taxpy(n, -step_size, grad, 1, x, 1);
	
}

/*	Update 's' correction vector

	If there's a curvature threshold, will also create a backup of the correction pair currently
	sitting in the memory slot into which the new pair will be written.

	Note that this procedure will not copy the new average into the previous average array,
	which needs to be done after updating 'y' in the main optimization function.
	
	x_sum					: sum of 'x' (optimization variables) since the last BFGS update
							  (will be overwritten during this procedure)
							  (pass 'x' for oLBFGS)
	x_avg_prev				: average values of 'x' during the interval of the previous BFGS update
							  (pass 'x_prev' for oLBFGS)
	n						: number of variables (dimensionality of 'x')
	needs_div				: whether x_sum should be divided to obtain the average
							  (pass 0 if it's already an average)
	bfgs_memory (in, out)	: BFGS memory struct
	nthreads				: number of parallel threads to use
*/
static inline void update_s_vector(real_t x_sum[], real_t x_avg_prev[], int n, int needs_div, bfgs_mem *bfgs_memory, int nthreads)
{
	/*	oLBFGS:	s = x - x_prev ----not computed here
		others:	s = x_avg - x_avg_prev
	*/
	backup_corr_pair(bfgs_memory, n, nthreads);
	if (needs_div) {  average_from_sum(x_sum, bfgs_memory->upd_freq, n);  }
	/* x_sum has now become x_avg --- this is aliased by the preprocessor, so don't worry about it not being declared */
	difference_elemwise(bfgs_memory->s_mem + bfgs_memory->mem_st_ix * n, x_avg, x_avg_prev, n, nthreads);
}

/*	Check curvature

	See if the new correction pair meets a minimum curvature threshold.
	If it does, accept it (store it), and if not, restore back the old correction pair,
	which was backed-up during the 'update_s_vector' procedure.
	
	bfgs_memory (in, out)	: BFGS memory struct
	n						: number of variables (dimensionality of 'x')
	iter_info				: pointer to the indicator on encountered problems
	nthreads				: number of parallel threads to use
*/
static inline void check_min_curvature(bfgs_mem *bfgs_memory, int n, info_enum *iter_info, int nthreads)
{
	/* s^T * y / s^T * s  >  epsilon */
	real_t *s = bfgs_memory->s_mem + bfgs_memory->mem_st_ix * n;;
	real_t *y = bfgs_memory->y_mem + bfgs_memory->mem_st_ix * n;
	real_t curv;

	if (bfgs_memory->min_curvature > 0)
	{
		curv = cblas_tdot(n, s, 1, y, 1) / cblas_tdot(n, s, 1, s, 1);
		if (curv <= bfgs_memory->min_curvature)
		{
			rollback_corr_pair(bfgs_memory, n, iter_info, nthreads);
			return;
		}
	}
	incr_bfgs_counters(bfgs_memory);
}

/*	Update 'y' correction vector using gradient differences

	Note: 'x_sum' needs to be reset after this (SQN and adaQN)
	
	grad 					: gradient (at new 'x' on the same batch for oLBFGS, at 'x_avg' on a larger batch for others)
	grad_prev				: previous gradient (at previous 'x' for oLBFGS, at 'x_avg_prev' on the previous large batch for others)
	bfgs_memory (in, out)	: BFGS memory struct
	n 						: number of variables (dimensionality of 'x')
	y_reg					: regularization parameter (will add this times 's' to 'y')
							  (pass 0 for SQN and adaQN)
	iter_info				: pointer to the indicator on encountered problems
	nthreads 				: number of parallel threads to use
*/
static inline void update_y_grad_diff(real_t grad[], real_t grad_prev[], bfgs_mem *bfgs_memory, int n, info_enum *iter_info, int nthreads)
{
	/*	oLBFGS:	y = grad_batch(x) - grad_batch(x_prev) + lambda * s
		others:	y = grad(x_avg) - grad_prev(x_avg_prev)
	*/
	real_t *s = bfgs_memory->s_mem + bfgs_memory->mem_st_ix * n;;
	real_t *y = bfgs_memory->y_mem + bfgs_memory->mem_st_ix * n;
	difference_elemwise(y, grad, grad_prev, n, nthreads);
	if (bfgs_memory->y_reg > 0){  cblas_taxpy(n, bfgs_memory->y_reg, s, 1, y, 1);  }

	check_min_curvature(bfgs_memory, n, iter_info, nthreads);
}

/*	Update 'y' correction vector using empirical Fisher matrix (adaQN)
	
	fisher_memory			: empirical Fisher struct
	bfgs_memory (in, out)	: BFGS memory struct
	n 						: number of variables (dimensionality of 'x')
	iter_info				: pointer to the indicator on encountered problems
	nthreads				: number of parallel threads to use
*/
static inline void update_y_fisher(fisher_mem *fisher_memory, bfgs_mem *bfgs_memory, int n, info_enum *iter_info, int nthreads)
{
	/* y = F' (F * s) / |F| */
	real_t *s = bfgs_memory->s_mem + bfgs_memory->mem_st_ix * n;
	real_t *y = bfgs_memory->y_mem + bfgs_memory->mem_st_ix * n;
	
	CBLAS_ORDER c_ord = CblasRowMajor;
	CBLAS_TRANSPOSE trans_no = CblasNoTrans;
	CBLAS_TRANSPOSE trans_yes = CblasTrans;

	cblas_tgemv(c_ord, trans_no, fisher_memory->mem_used, n, 1,
		fisher_memory->F, n, s, 1, 0, fisher_memory->buffer_y, 1);
	cblas_tgemv(c_ord, trans_yes, fisher_memory->mem_used, n, 1 / (real_t) fisher_memory->mem_used,
		fisher_memory->F, n, fisher_memory->buffer_y, 1, 0, y, 1);

	check_min_curvature(bfgs_memory, n, iter_info, nthreads);
}

/*	Update 'y' correction vector using the production between the Hessian and the 's' vector

	hess_vec				: calculated Hessian * s
	bfgs_memory (in, out)	: BFGS memory struct
	iter_info				: pointer to the indicator on encountered problems
	n 						: number of variables (dimensionality of 'x')
	nthreads				: number of parallel threads to use
*/
static inline void update_y_hessvec(real_t hess_vec[], bfgs_mem *bfgs_memory, info_enum *iter_info, int n, int nthreads)
{
	copy_arr(hess_vec, bfgs_memory->y_mem + bfgs_memory->mem_st_ix * n, n, nthreads);
	check_min_curvature(bfgs_memory, n, iter_info, nthreads);
}

/*	============= Optimizer functions for the external API =============
	
	Documentation for them can be found in the header file.

	These functions are very hard to follow, but think of them like this:
	each of them will send you to a different part as if it were a 'goto',
	only there will be an interruption in between where the required calculation
	is requested externally. Check which part sent you to where you currently are,
	and where is each part going to send you next.
*/
int run_oLBFGS(real_t step_size, real_t x[], real_t grad[], real_t **req, task_enum *task, workspace_oLBFGS *oLBFGS, info_enum *iter_info)
{
	*iter_info = no_problems_encountered;

	/* first run: immediately request a gradient */
	if (oLBFGS->section == 0)
	{
		*task = calc_grad;
		*req = x;
		oLBFGS->section = 1;
		return 0;
	}

	/* second run (main loop): save grad, take a step, save delta_x, request another gradient in same batch */
	if (oLBFGS->section == 1)
	{

		/* save gradient */
		copy_arr(grad, oLBFGS->grad_prev, oLBFGS->n, oLBFGS->nthreads);

		/* take a step */
		take_step(step_size, oLBFGS->n, x, grad, oLBFGS->bfgs_memory, 0,
			NULL, oLBFGS->hess_init, NULL, 0, oLBFGS->check_nan, iter_info, oLBFGS->nthreads);
		oLBFGS->niter++;

		/* store differences in BFGS memory */
		if (*iter_info == no_problems_encountered){
			backup_corr_pair(oLBFGS->bfgs_memory, oLBFGS->n, oLBFGS->nthreads); /* rollback happens on 'update_y_grad_diff' */
			cblas_tscal(oLBFGS->n, -step_size, grad, 1);
			copy_arr(grad, oLBFGS->bfgs_memory->s_mem + oLBFGS->bfgs_memory->mem_st_ix * oLBFGS->n, oLBFGS->n, oLBFGS->nthreads);

			/* request another gradient */
			*task = calc_grad_same_batch;
			*req = x;
			oLBFGS->section = 2;
			return 1;
		} else {
			if (*iter_info == search_direction_was_nan) {  flush_bfgs_mem(oLBFGS->bfgs_memory);  }
			*task = calc_grad;
			*req = x;
			oLBFGS->section = 1;
			return 0;
		}
	}

	/* third run (loop): update correction pairs, request a gradient on new batch */
	if (oLBFGS->section == 2)
	{
		update_y_grad_diff(grad, oLBFGS->grad_prev, oLBFGS->bfgs_memory, oLBFGS->n, iter_info, oLBFGS->nthreads);
		*task = calc_grad;
		*req = x;
		oLBFGS->section = 1;
		return 0;
	}

	*task = invalid_input;
	fprintf(stderr, "oLBFGS got an invalid workspace as input.\n");
	return -1000;
}

int run_SQN(real_t step_size, real_t x[], real_t grad[], real_t hess_vec[], real_t **req, real_t **req_vec, task_enum *task, workspace_SQN *SQN, info_enum *iter_info)
{
	*iter_info = no_problems_encountered;
	int return_value = 0;

	/* first run: immediately request a gradient */
	if (SQN->section == 0)
	{
		// add_to_sum(x, SQN->x_sum, SQN->n, SQN->nthreads);
		goto resume_main_loop;
	}

	/* second run (main loop): take a step, save sum, see if it's time for creating correction pair */
	if (SQN->section == 1)
	{

		/* take a step */
		take_step(step_size, SQN->n, x, grad, SQN->bfgs_memory, 0, NULL, 0, NULL, 0,
			SQN->check_nan, iter_info, SQN->nthreads);
		SQN->niter++;

		/* check for unchanged parameters */
		if (*iter_info == search_direction_was_nan) {return_value = 0;}
		else {return_value = 1;}

		/*	save sum of new values
			note: even if they are not updated, need to maintain the sum in the same magnitude,
			as it will be divided by L
		*/
		add_to_sum(x, SQN->x_sum, SQN->n, SQN->nthreads);

		/* usually, requests a new gradient and returns right here */
		if ( (SQN->niter % SQN->bfgs_memory->upd_freq) != 0 )
		{
			goto resume_main_loop;
		}

		/* at some intervals, update hessian approx */

		/* exception: the first time, just store the averages - if using grad diff, request a long gradient on those, else go back */
		if (SQN->niter == SQN->bfgs_memory->upd_freq)
		{
			average_from_sum(SQN->x_sum, SQN->bfgs_memory->upd_freq, SQN->n);
			archive_x_avg(SQN->x_avg, SQN->x_avg_prev, SQN->n, SQN->nthreads);
			/* note: x_avg is alised by the preprocessor as synonym to x_sum */

			if (SQN->use_grad_diff)
			{
				*task = calc_grad_big_batch;
				*req = SQN->x_avg_prev;
				SQN->section = 2;
				return return_value;
			}
			else {
				goto resume_main_loop;
			}
		}

		/* first update 's' (turns the sum to avg), but don't reset the sum yet as it'll be needed for a hessian-vec or long grad */
		update_s_vector(SQN->x_sum, SQN->x_avg_prev, SQN->n, 1, SQN->bfgs_memory, SQN->nthreads);

		/* request long grad on the new average */
		if (SQN->use_grad_diff)
		{
			*task = calc_grad_big_batch;
			SQN->section = 3;
			*req = SQN->x_avg;
		}
		/* request hessian-vector on the differences between the averages */
		else
		{
			*task = calc_hess_vec;
			SQN->section = 4;
			*req = SQN->x_avg;
			*req_vec = SQN->bfgs_memory->s_mem + SQN->n * SQN->bfgs_memory->mem_st_ix;
		}
		return return_value;
	}

	/* third run: got a long gradient on first averages, store it and go back */
	if (SQN->section == 2)
	{
		copy_arr(grad, SQN->grad_prev, SQN->n, SQN->nthreads);
		goto resume_main_loop;
	}

	/* fourth run (loop): got a long gradient on new averages, reset sum, create correction pair and go back */
	if (SQN->section == 3)
	{
		update_y_grad_diff(grad, SQN->grad_prev, SQN->bfgs_memory, SQN->n, iter_info, SQN->nthreads);
		if (*iter_info == no_problems_encountered){
			copy_arr(grad, SQN->grad_prev, SQN->n, SQN->nthreads);
			copy_arr(SQN->x_avg, SQN->x_avg_prev, SQN->n, SQN->nthreads);
		}
		set_to_zero(SQN->x_sum, SQN->n, SQN->nthreads);
		goto resume_main_loop;
	}

	/* fifth run (loop): got a hessian-vector product, reset sum, create a correction pair and go back */
	if (SQN->section == 4)
	{
		archive_x_avg(SQN->x_avg, SQN->x_avg_prev, SQN->n, SQN->nthreads);
		update_y_hessvec(hess_vec, SQN->bfgs_memory, iter_info, SQN->n, SQN->nthreads);
		goto resume_main_loop;
	}

	*task = invalid_input;
	fprintf(stderr, "SQN got an invalid workspace as input.\n");
	return -1000;

	resume_main_loop:
		SQN->section = 1;
		*task = calc_grad;
		*req = x;
		return return_value;
}

int run_adaQN(real_t step_size, real_t x[], real_t f, real_t grad[], real_t **req, task_enum *task, workspace_adaQN *adaQN, info_enum *iter_info)
{
	*iter_info = no_problems_encountered;
	int return_value = 0;

	/* first run: immediately request a gradient */
	if (adaQN->section == 0)
	{
		// add_to_sum(x, adaQN->x_sum, adaQN->n, adaQN->nthreads);
		goto resume_main_loop;
	}

	/*	second run (main loop): store gradient, take a step (gradient_sq is summed there), sum x,
		see if it's time for creating correction pair --if so, request either long grad or function
	*/
	if (adaQN->section == 1)
	{

		/* store gradient */
		add_to_fisher_mem(grad, adaQN->fisher_memory, adaQN->n, adaQN->nthreads);

		/* take a step */
		take_step(step_size, adaQN->n, x, grad, adaQN->bfgs_memory, adaQN->rmsprop_weight,
			adaQN->H0, 0, adaQN->grad_sum_sq, adaQN->scal_reg, adaQN->check_nan, iter_info, adaQN->nthreads);
		if (*iter_info == search_direction_was_nan)
		{
			// flush_fisher_mem(adaQN->fisher_memory);
			return_value = 0;
		}
		else { return_value = 1; }
		adaQN->niter++;

		/*	save sum of new values
			note: even if they are not updated, need to maintain the sum in the same magnitude,
			as it will be divided by L
		*/
		add_to_sum(x, adaQN->x_sum, adaQN->n, adaQN->nthreads);

		/* usually, requests a new gradient and returns right here */
		if ( (adaQN->niter % adaQN->bfgs_memory->upd_freq) != 0 )
		{
			goto resume_main_loop;
		}

		/* at some intervals, update hessian approx */

		/*	exception: the first time, just store the averages, then:
			-if use_grad_diff, request a long gradient on the averages (function comes later)
			-if using max_incr, request a function on the averages
			-if neither, go back to main loop
		*/
		if (adaQN->niter == adaQN->bfgs_memory->upd_freq)
		{
			average_from_sum(adaQN->x_sum, adaQN->bfgs_memory->upd_freq, adaQN->n);
			archive_x_avg(adaQN->x_avg, adaQN->x_avg_prev, adaQN->n, adaQN->nthreads);
			/* note: x_avg is aliased by the preprocessor as synonym to x_sum */
			if (adaQN->use_grad_diff){
				*task = calc_grad_big_batch;
				*req = adaQN->x_avg_prev;
				adaQN->section = 2;
				return return_value;
			}
			if (adaQN->max_incr > 0){
				*task = calc_fun_val_batch;
				*req = adaQN->x_avg_prev;
				adaQN->section = 3;
				return return_value;
			}
			goto resume_main_loop;
		}

		/* evaluate function on new averages if needed */
		if (adaQN->max_incr > 0)
		{
			average_from_sum(adaQN->x_sum, adaQN->bfgs_memory->upd_freq, adaQN->n);
			*task = calc_fun_val_batch;
			*req = adaQN->x_avg;
			adaQN->section = 5;
			return return_value;
		}

		/* first update 's' (turns the sum to avg), but don't reset the sum yet as it'll be needed for a hessian-vec or long grad */
		update_s_vector(adaQN->x_sum, adaQN->x_avg_prev, adaQN->n, 1, adaQN->bfgs_memory, adaQN->nthreads);
		goto update_y;
	}

	/* third run: got a long gradient on first averages, store it and go back */
	if (adaQN->section == 2)
	{
		copy_arr(grad, adaQN->grad_prev, adaQN->n, adaQN->nthreads);

		/* ask for function if needed */
		if (adaQN->max_incr){
			*task = calc_fun_val_batch;
			*req = adaQN->x_avg_prev;
			adaQN->section = 3;
			return 0;
		} else {
			goto resume_main_loop;
		}
	}

	/* fourth run: got first function eval on validation batch, store it and request a gradient */
	if (adaQN->section == 3)
	{
		adaQN->f_prev = f;
		goto resume_main_loop;
	}

	/* fifth run (loop): got a long gradient on new averages, create correction pair (function was asked before) */
	if (adaQN->section == 4){
		update_y_grad_diff(grad, adaQN->grad_prev, adaQN->bfgs_memory, adaQN->n, iter_info, adaQN->nthreads);
		if (*iter_info == no_problems_encountered) {  copy_arr(grad, adaQN->grad_prev, adaQN->n, adaQN->nthreads);  }
		set_to_zero(adaQN->x_sum, adaQN->n, adaQN->nthreads);
		goto resume_main_loop;
	}

	/* sixth run (loop): evaluated function on new averages, now see whether to keep correction pair */
	if (adaQN->section == 5)
	{
		if (f > adaQN->max_incr * adaQN->f_prev || isinf(f) || isnan(f) )
		{
			flush_bfgs_mem(adaQN->bfgs_memory);
			flush_fisher_mem(adaQN->fisher_memory);
			copy_arr(adaQN->x_avg_prev, x, adaQN->n, adaQN->nthreads);
			*iter_info = func_increased;
			return_value = 1;
			goto resume_main_loop;
		}

		else
		{
			adaQN->f_prev = f;
			update_s_vector(adaQN->x_avg, adaQN->x_avg_prev, adaQN->n, 0, adaQN->bfgs_memory, adaQN->nthreads);
			goto update_y;
		}
	}

	*task = invalid_input;
	fprintf(stderr, "adaQN got an invalid workspace as input.\n");
	return -1000;

	update_y:
		if (adaQN->use_grad_diff) {
			*req = adaQN->x_avg;
			*task = calc_grad_big_batch;
			adaQN->section = 4;
			return return_value;
		} else {
			update_y_fisher(adaQN->fisher_memory, adaQN->bfgs_memory, adaQN->n, iter_info, adaQN->nthreads);
			if (*iter_info == no_problems_encountered) { copy_arr(adaQN->x_avg, adaQN->x_avg_prev, adaQN->n, adaQN->nthreads); }
			set_to_zero(adaQN->x_sum, adaQN->n, adaQN->nthreads);
			goto resume_main_loop;
		}

	resume_main_loop:
		adaQN->section = 1;
		*task = calc_grad;
		*req = x;
		return return_value;
}

#ifdef __cplusplus
}
#endif
