#' @useDynLib stochQN
#' @importFrom parallel detectCores
#' @importFrom stats predict
#' @importFrom stats coef
#' @importFrom stats model.matrix
#' @importFrom stats rnorm
#' @importFrom stats runif
#' @importFrom stats terms
NULL

#' @title oLBFGS Free-Mode Optimizer
#' @description Optimizes an empirical (convex) loss function over batches of sample data. Compared to
#' function/class 'oLBFGS', this version lets the user do all the calculations from the outside, only
#' interacting with the object by means of a function that returns a request type and is fed the
#' required calculation through a method 'update_gradient'.
#' 
#' Order in which requests are made:
#' 	
#' 	========== loop ===========
#' 	
#' 	* calc_grad
#' 	
#'  * calc_grad_same_batch		(might skip if using check_nan)
#'  
#'  ===========================
#'  
#' After running this function, apply `run_oLBFGS_free` to it to get the first requested piece of information.
#' @param mem_size Number of correction pairs to store for approximation of Hessian-vector products.
#' @param hess_init Value to which to initialize the diagonal of H0.
#' If passing `NULL`, will use the same initializion as for SQN ((s_last * y_last) / (y_last * y_last)).
#' @param min_curvature Minimum value of (s * y) / (s * s) in order to accept a correction pair. Pass `NULL` for
#' no minimum.
#' @param y_reg Regularizer for 'y' vector (gets added y_reg * s). Pass `NULL` for no regularization.
#' @param check_nan Whether to check for variables becoming NA after each iteration, and reverting the step if they do
#' (will also reset BFGS memory).
#' @param nthreads Number of parallel threads to use. If set to -1, will determine the number of available threads and use
#' all of them. Note however that not all the computations can be parallelized.
#' @return An `oLBFGS_free` object, which can be used through functions `update_gradient` and `run_oLBFGS_free`
#' @seealso \link{update_gradient} , \link{run_oLBFGS_free}
#' @references \itemize{ \item Schraudolph, N.N., Yu, J. and Guenter, S., 2007, March.
#' "A stochastic quasi-Newton method for online convex optimization."
#' In Artificial Intelligence and Statistics (pp. 436-443).
#' \item Wright, S. and Nocedal, J., 1999. "Numerical optimization." (ch 7) Springer Science, 35(67-68), p.7.}
#' @examples 
#' ### Example optimizing Rosenbrock 2D function
#' ### Note that this example is not stochastic, as the
#' ### function is not evaluated in expectation based on
#' ### batches of data, but rather it has a given absolute
#' ### form that never varies.
#' ### Warning: this optimizer is meant for convex functions
#' ### (Rosenbrock's is not convex)
#' library(stochQN)
#' 
#' 
#' fr <- function(x) { ## Rosenbrock Banana function
#' 	x1 <- x[1]
#' 	x2 <- x[2]
#' 	100 * (x2 - x1 * x1)^2 + (1 - x1)^2
#' }
#' grr <- function(x) { ## Gradient of 'fr'
#' 	x1 <- x[1]
#' 	x2 <- x[2]
#' 	c(-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1),
#' 	200 * (x2 - x1 * x1))
#' }
#' 
#' ### Initial values of x
#' x_opt = as.numeric(c(0, 2))
#' cat(sprintf("Initial values of x: [%.3f, %.3f]\n",
#' 	x_opt[1], x_opt[2]))
#' 	
#' ### Will use a constant step size throughout
#' ### (not recommended)
#' step_size <- 1e-1
#' 
#' ### Initialize the optimizer
#' optimizer <- oLBFGS_free()
#' 
#' ### Keep track of the iteration number
#' curr_iter <- 0
#' 
#' ### Run a loop for 100 iterations
#' ### (Note that each iteration requires 2 calculations,
#' ###  hence the 200)
#' for (i in 1:200) {
#' 	req <- run_oLBFGS_free(optimizer, x_opt, step_size)
#' 	if (req$task == "calc_grad") {
#' 	  update_gradient(optimizer, grr(req$requested_on))
#' 	} else if (req$task == "calc_grad_same_batch") {
#' 	  update_gradient(optimizer, grr(req$requested_on))
#' 	}
#' 	
#' 	### Track progress every 10 iterations
#' 	if (req$info$iteration_number > curr_iter) {
#' 	  curr_iter <- req$info$iteration_number
#' 	  if ((curr_iter %% 10) == 0) {
#' 	  cat(sprintf(
#' 	   "Iteration %3d - Current function value: %.3f\n",
#' 	  req$info$iteration_number, fr(x_opt)
#' 	  ))
#' 	  }
#' 	}
#' }
#' cat(sprintf("Current values of x: [%.3f, %.3f]\n",
#' 	x_opt[1], x_opt[2]))
#' @export
oLBFGS_free <- function(mem_size = 10, hess_init = NULL, min_curvature = 1e-4,
						y_reg = NULL,check_nan = TRUE, nthreads = -1) {
	this <- list(saved_params = list())
	this$saved_params <- take.common.inputs(this$saved_params, mem_size, min_curvature, y_reg, check_nan, nthreads)
	if (!is.null(hess_init)) {
		this$saved_params$hess_init <- check.positive.float(hess_init, "hess_init")
	} else {
		this$saved_params$hess_init <- as.numeric(0)
	}
	this$initialized <- FALSE
	class(this) <- "oLBFGS_free"
	return(this)
}

#' @title SQN Free-Mode Optimizer
#' @description Optimizes an empirical (convex) loss function over batches of sample data. Compared to
#' function/class 'SQN', this version lets the user do all the calculations from the outside, only
#' interacting with the object by means of a function that returns a request type and is fed the
#' required calculation through methods 'update_gradient' and 'update_hess_vec'.
#' 
#' Order in which requests are made:
#' 	
#' 	========== loop ===========
#' 	
#' 	* calc_grad
#' 	
#' \verb{   }... (repeat calc_grad)
#'   
#' if 'use_grad_diff':
#' 
#' \verb{    }* calc_grad_big_batch
#' 	
#' else:
#' 
#' \verb{    }* calc_hess_vec
#' 	
#' ===========================
#' 
#' After running this function, apply `run_SQN_free` to it to get the first requested piece of information.
#' @param mem_size Number of correction pairs to store for approximation of Hessian-vector products.
#' @param bfgs_upd_freq Number of iterations (batches) after which to generate a BFGS correction pair.
#' @param min_curvature Minimum value of (s * y) / (s * s) in order to accept a correction pair. Pass `NULL` for
#' no minimum.
#' @param y_reg Regularizer for 'y' vector (gets added y_reg * s). Pass `NULL` for no regularization.
#' @param use_grad_diff Whether to create the correction pairs using differences between gradients instead of Hessian-vector products.
#' These gradients are calculated on a larger batch than the regular ones (given by batch_size * bfgs_upd_freq).
#' @param check_nan Whether to check for variables becoming NaN after each iteration, and reverting the step if they do
#' (will also reset BFGS memory).
#' @param nthreads Number of parallel threads to use. If set to -1, will determine the number of available threads and use
#' all of them. Note however that not all the computations can be parallelized.
#' @return An `SQN_free` object, which can be used through functions `update_gradient`, `update_hess_vec`,
#' and `run_SQN_free`
#' @seealso \link{update_gradient} , \link{update_hess_vec} , \link{run_oLBFGS_free}
#' @references \itemize{ \item Byrd, R.H., Hansen, S.L., Nocedal, J. and Singer, Y., 2016.
#' "A stochastic quasi-Newton method for large-scale optimization."
#' SIAM Journal on Optimization, 26(2), pp.1008-1031.
#' \item Wright, S. and Nocedal, J., 1999. "Numerical optimization." (ch 7) Springer Science, 35(67-68), p.7.}
#' @examples 
#' ### Example optimizing Rosenbrock 2D function
#' ### Note that this example is not stochastic, as the
#' ### function is not evaluated in expectation based on
#' ### batches of data, but rather it has a given absolute
#' ### form that never varies.
#' ### Warning: this optimizer is meant for convex functions
#' ### (Rosenbrock's is not convex)
#' library(stochQN)
#' 
#' 
#' fr <- function(x) { ## Rosenbrock Banana function
#' 	x1 <- x[1]
#' 	x2 <- x[2]
#' 	100 * (x2 - x1 * x1)^2 + (1 - x1)^2
#' }
#' grr <- function(x) { ## Gradient of 'fr'
#' 	x1 <- x[1]
#' 	x2 <- x[2]
#' 	c(-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1),
#' 	  200 * (x2 - x1 * x1))
#' }
#' Hvr <- function(x, v) { ## Hessian of 'fr' by vector 'v'
#' 	x1 <- x[1]
#' 	x2 <- x[2]
#' 	H <- matrix(c(1200 * x1^2 - 400*x2 + 2,
#' 			  -400 * x1, -400 * x1, 200),
#' 			nrow = 2)
#' 	as.vector(H %*% v)
#' }
#' 
#' ### Initial values of x
#' x_opt = as.numeric(c(0, 2))
#' cat(sprintf("Initial values of x: [%.3f, %.3f]\n",
#' 			x_opt[1], x_opt[2]))
#' 
#' ### Will use constant step size throughout
#' ### (not recommended)
#' step_size = 1e-3
#' 
#' ### Initialize the optimizer
#' optimizer = SQN_free()
#' 
#' ### Keep track of the iteration number
#' curr_iter <- 0
#' 
#' ### Run a loop for severa, iterations
#' ### (Note that some iterations might require more
#' ###  than 1 calculation request)
#' for (i in 1:200) {
#'   req <- run_SQN_free(optimizer, x_opt, step_size)
#'   if (req$task == "calc_grad") {
#'     update_gradient(optimizer, grr(req$requested_on$req_x))
#'   } else if (req$task == "calc_hess_vec") {
#' 	   update_hess_vec(optimizer,
#'       Hvr(req$requested_on$req_x, req$requested_on$req_vec))
#'   }
#' 
#'   ### Track progress every 10 iterations
#'   if (req$info$iteration_number > curr_iter) {
#'   	curr_iter <- req$info$iteration_number
#'   }
#'   if ((curr_iter %% 10) == 0) {
#'   	cat(sprintf(
#'   	 "Iteration %3d - Current function value: %.3f\n",
#'   	 req$info$iteration_number, fr(x_opt)))
#'   }
#' }
#' cat(sprintf("Current values of x: [%.3f, %.3f]\n",
#' 			x_opt[1], x_opt[2]))
#' @export
SQN_free <- function(mem_size = 10, bfgs_upd_freq = 20, min_curvature = 1e-4, y_reg = NULL,
					 use_grad_diff = FALSE, check_nan = TRUE, nthreads = -1) {
	this <- list(saved_params = list())
	this$saved_params <- take.common.inputs(this$saved_params, mem_size, min_curvature, y_reg, check_nan, nthreads)
	this$saved_params$bfgs_upd_freq <- check.positive.integer(bfgs_upd_freq, "bfgs_upd_freq")
	this$saved_params$use_grad_diff <- check.is.bool(use_grad_diff, "use_grad_diff")
	this$initialized <- FALSE
	class(this) <- "SQN_free"
	return(this)
}

#' @title adaQN Free-Mode Optimizer
#' @description Optimizes an empirical (perhaps non-convex) loss function over batches of sample data. Compared to
#' function/class 'adaQN', this version lets the user do all the calculations from the outside, only
#' interacting with the object by means of a function that returns a request type and is fed the
#' required calculation through methods 'update_gradient' and 'update_function'.
#' 
#' Order in which requests are made:
#' 	
#' 	========== loop ===========
#' 	
#' 	* calc_grad
#' 	
#' \verb{    }... (repeat calc_grad)
#' 
#' if max_incr > 0:
#' 
#' \verb{    }* calc_fun_val_batch
#' 	
#' if 'use_grad_diff':
#' 
#' \verb{    }* calc_grad_big_batch	(skipped if below max_incr)
#' 	
#' ===========================
#' 
#' After running this function, apply `run_adaQN_free` to it to get the first requested piece of information.
#' @param mem_size Number of correction pairs to store for approximation of Hessian-vector products.
#' @param fisher_size Number of gradients to store for calculation of the empirical Fisher product with gradients.
#' If passing `NULL`, will force `use_grad_diff` to `TRUE`.
#' @param bfgs_upd_freq Number of iterations (batches) after which to generate a BFGS correction pair.
#' @param max_incr Maximum ratio of function values in the validation set under the average values of `x` during current epoch
#' vs. previous epoch. If the ratio is above this threshold, the BFGS and Fisher memories will be reset, and `x`
#' values reverted to their previous average.
#' Pass `NULL` for no function-increase checking.
#' @param min_curvature Minimum value of (s * y) / (s * s) in order to accept a correction pair. Pass `NULL` for
#' no minimum.
#' @param scal_reg Regularization parameter to use in the denominator for AdaGrad and RMSProp scaling.
#' @param rmsprop_weight If not `NULL`, will use RMSProp formula instead of AdaGrad for approximated inverse-Hessian initialization.
#' @param y_reg Regularizer for 'y' vector (gets added y_reg * s). Pass `NULL` for no regularization.
#' @param use_grad_diff Whether to create the correction pairs using differences between gradients instead of Fisher matrix.
#' These gradients are calculated on a larger batch than the regular ones (given by batch_size * bfgs_upd_freq).
#' If `TRUE`, empirical Fisher matrix will not be used.
#' @param check_nan Whether to check for variables becoming NaN after each iteration, and reverting the step if they do
#' (will also reset BFGS and Fisher memory).
#' @param nthreads Number of parallel threads to use. If set to -1, will determine the number of available threads and use
#' all of them. Note however that not all the computations can be parallelized.
#' @return An `adaQN_free` object, which can be used through functions `update_gradient`, `update_fun`, and `run_adaQN_free`
#' @seealso \link{update_gradient} , \link{update_fun} , \link{run_adaQN_free}
#' @references \itemize{ \item Keskar, N.S. and Berahas, A.S., 2016, September.
#' "adaQN: An Adaptive Quasi-Newton Algorithm for Training RNNs."
#' In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 1-16). Springer, Cham.
#' \item Wright, S. and Nocedal, J., 1999. "Numerical optimization." (ch 7) Springer Science, 35(67-68), p.7.}
#' @examples 
#' ### Example optimizing Rosenbrock 2D function
#' ### Note that this example is not stochastic, as the
#' ### function is not evaluated in expectation based on
#' ### batches of data, but rather it has a given absolute
#' ### form that never varies.
#' library(stochQN)
#' 
#' 
#' fr <- function(x) { ## Rosenbrock Banana function
#' 	x1 <- x[1]
#' 	x2 <- x[2]
#' 	100 * (x2 - x1 * x1)^2 + (1 - x1)^2
#' }
#' grr <- function(x) { ## Gradient of 'fr'
#' 	x1 <- x[1]
#' 	x2 <- x[2]
#' 	c(-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1),
#' 	  200 * (x2 - x1 * x1))
#' }
#' 
#' 
#' ### Initial values of x
#' x_opt = as.numeric(c(0, 2))
#' cat(sprintf("Initial values of x: [%.3f, %.3f]\n",
#' 			x_opt[1], x_opt[2]))
#' 
#' ### Will use constant step size throughout
#' ### (not recommended)
#' step_size = 1e-2
#' 
#' ### Initialize the optimizer
#' optimizer = adaQN_free()
#' 
#' ### Keep track of the iteration number
#' curr_iter <- 0
#' 
#' ### Run a loop for many iterations
#' ### (Note that some iterations might require more
#' ###  than 1 calculation request)
#' for (i in 1:200) {
#' 	req <- run_adaQN_free(optimizer, x_opt, step_size)
#' 	if (req$task == "calc_grad") {
#' 	  update_gradient(optimizer, grr(req$requested_on))
#' 	} else if (req$task == "calc_fun_val_batch") {
#' 	  update_fun(optimizer, fr(req$requested_on))
#' 	}
#' 
#' 	### Track progress every 10 iterations
#' 	if (req$info$iteration_number > curr_iter) {
#' 		curr_iter <- req$info$iteration_number
#' 	}
#' 	if ((curr_iter %% 10) == 0) {
#' 		cat(sprintf(
#' 		  "Iteration %3d - Current function value: %.3f\n",
#' 		  req$info$iteration_number, fr(x_opt)))
#' 	}
#' }
#' cat(sprintf("Current values of x: [%.3f, %.3f]\n",
#' 			x_opt[1], x_opt[2]))
#' @export
adaQN_free <- function(mem_size = 10, fisher_size = 100, bfgs_upd_freq = 20, max_incr = 1.01,
					   min_curvature = 1e-4, scal_reg = 1e-4, rmsprop_weight = 0.9,
					   y_reg = NULL, use_grad_diff = FALSE, check_nan = TRUE, nthreads = -1) {
	this <- list(saved_params = list())
	this$saved_params <- take.common.inputs(this$saved_params, mem_size, min_curvature, y_reg, check_nan, nthreads)
	if (is.null(fisher_size)) {
		fisher_size   <- 1
		use_grad_diff <- TRUE
	}
	if (use_grad_diff) {
		fisher_size   <- 1
	}
	
	this$saved_params$bfgs_upd_freq <- check.positive.integer(bfgs_upd_freq, "bfgs_upd_freq")
	this$saved_params$use_grad_diff <- check.is.bool(use_grad_diff, "use_grad_diff")
	this$saved_params$fisher_size   <- check.positive.integer(fisher_size, "fisher_size")
	this$saved_params$scal_reg      <- check.positive.float(scal_reg, "scal_reg")
	if (is.null(rmsprop_weight)) {
		this$saved_params$rmsprop_weight <- as.numeric(0)
	} else {
		this$saved_params$rmsprop_weight <- check.positive.float(rmsprop_weight, "rmsprop_weight")
		if ((this$saved_params$rmsprop_weight <= 0) | (this$saved_params$rmsprop_weight >= 1)) {
			stop("'rmsprop_weight' must be between zero and one.")
		}
	}
	if (is.null(max_incr)) {
		this$saved_params$max_incr <- as.numeric(0)
	} else {
		this$saved_params$max_incr <- check.positive.float(max_incr, "max_incr")
	}
	this$initialized <- FALSE
	class(this) <- "adaQN_free"
	return(this)
}

#' @title Run oLBFGS optimizer in free-mode
#' @description Run the next step of an oLBFGS optimization procedure, after the last requested calculation
#' has been fed to the optimizer. When run for the first time, there is no request, so the function just
#' needs to be run on the object as it is returned from function `oLBFGS_free`.
#' @param optimizer An `oLBFGS_free` optimizer, for which its last request must have been served. Will be updated in-place.
#' @param x Current values of the variables being optimized. Must be a numeric vector. Will be updated in-place.
#' @param step_size Step size for the quasi-Newton update.
#' @return A request with the next piece of required information. The output will be a list with the following levels:
#' \itemize{
#'    \item{task} Requested task (one of "calc_grad" or "calc_grad_same_batch").
#'    \item{requested_on} Values of `x` at which the requested information must be calculated.
#'    \item{info} \itemize{
#'       \item{x_changed_in_run} Whether the `x` vector was updated.
#'       \item{iteration_number} Current iteration number (in terms of quasi-Newton updates).
#'       \item{iteration_info} Information about potential problems encountered during the iteration.
#'    } 
#' }
#' @export
#' @seealso \link{oLBFGS_free}
run_oLBFGS_free <- function(optimizer, x, step_size) {
	check.x.and.step.size(x, step_size)
	if (!("oLBFGS_free" %in% class(optimizer))) stop("This function only applies to free-mode oLBFGS optimizer.")
	
	if (!optimizer$initialized) {
		oLBFGS_obj <- create.r.oLBFGS(NROW(x), optimizer$saved_params$mem_size, optimizer$saved_params$hess_init,
									  optimizer$saved_params$y_reg, optimizer$saved_params$min_curvature,
									  optimizer$saved_params$check_nan, optimizer$saved_params$nthreads)
		eval.parent(substitute(optimizer[["oLBFGS"]] <- oLBFGS_obj))
		eval.parent(substitute(optimizer[["initialized"]]  <- TRUE))
		eval.parent(substitute(optimizer[["saved_params"]] <- NULL))
		grad_init <- vector(mode = "numeric", length =  oLBFGS_obj$n)
		eval.parent(substitute(optimizer[["gradient"]] <- grad_init))
		optimizer$oLBFGS   <- oLBFGS_obj
		optimizer$gradient <- grad_init
		return(list(
			task = get.task(101),
			requested_on = x,
			info = list(
				x_changed_in_run = get.x.changed(0),
				iteration_number = 0,
				iteration_info   = get.iter.info(200)
			)
		))
	}
	if (NROW(x) != optimizer$oLBFGS$n) stop("'x' has wrong dimensions.")
	
	req_vec   <- vector(mode = "numeric", length =  optimizer$oLBFGS$n)
	x_changed <- as.integer(0)
	iter_info <- as.integer(0)
	task      <- as.integer(0)
	.Call("r_run_oLBFGS", optimizer$oLBFGS$BFGS_mem$s_mem, optimizer$oLBFGS$BFGS_mem$y_mem,
		  optimizer$oLBFGS$BFGS_mem$buffer_rho, optimizer$oLBFGS$BFGS_mem$buffer_alpha,
		  optimizer$oLBFGS$BFGS_mem$s_bak, optimizer$oLBFGS$BFGS_mem$y_bak,
		  optimizer$oLBFGS$BFGS_mem$mem_size, optimizer$oLBFGS$BFGS_mem$mem_used,
		  optimizer$oLBFGS$BFGS_mem$mem_st_ix, optimizer$oLBFGS$BFGS_mem$upd_freq,
		  optimizer$oLBFGS$BFGS_mem$y_reg, optimizer$oLBFGS$BFGS_mem$min_curvature,
		  optimizer$oLBFGS$grad_prev, optimizer$oLBFGS$hess_init, optimizer$oLBFGS$niter,
		  optimizer$oLBFGS$section, optimizer$oLBFGS$nthreads, optimizer$oLBFGS$check_nan, optimizer$oLBFGS$n,
		  x, optimizer$gradient, step_size,
		  x_changed, req_vec, task, iter_info)
	return(list(
		task = get.task(task),
		requested_on = req_vec,
		info = list(
			x_changed_in_run = get.x.changed(x_changed),
			iteration_number = as.integer(optimizer$oLBFGS$niter) + 1 - 1,
			iteration_info   = get.iter.info(iter_info)
		)
	))
}

#' @title Run SQN optimizer in free-mode
#' @description Run the next step of an SQN optimization procedure, after the last requested calculation
#' has been fed to the optimizer. When run for the first time, there is no request, so the function just
#' needs to be run on the object as it is returned from function `SQN_free`.
#' @param optimizer An `SQN_free` optimizer, for which its last request must have been served. Will be updated in-place.
#' @param x Current values of the variables being optimized. Must be a numeric vector. Will be updated in-place.
#' @param step_size Step size for the quasi-Newton update.
#' @return A request with the next piece of required information. The output will be a list with the following levels:
#' \itemize{
#'    \item{task} Requested task (one of "calc_grad", "calc_grad_big_batch", "calc_hess_vec").
#'    \item{requested_on} \itemize{
#'        \item{req_x} Values of `x` at which the requested information (gradient/Hessian) must be calculated.
#'        \item{req_vec} Vector by which the Hessian must be multiplied. Will output `NULL` when this
#'        calculation is not needed.
#'    }
#'    \item{info} \itemize{
#'       \item{x_changed_in_run} Whether the `x` vector was updated.
#'       \item{iteration_number} Current iteration number (in terms of quasi-Newton updates).
#'       \item{iteration_info} Information about potential problems encountered during the iteration.
#'    } 
#' }
#' @export
#' @seealso \link{SQN_free}
run_SQN_free <- function(optimizer, x, step_size) {
	check.x.and.step.size(x, step_size)
	if (!("SQN_free" %in% class(optimizer))) stop("This function only applies to free-mode SQN optimizer.")
	
	if (!optimizer$initialized) {
		SQN_obj <- create.r.SQN(NROW(x), optimizer$saved_params$mem_size, optimizer$saved_params$bfgs_upd_freq,
								optimizer$saved_params$min_curvature, optimizer$saved_params$use_grad_diff,
								optimizer$saved_params$y_reg, optimizer$saved_params$check_nan,
								optimizer$saved_params$nthreads)
		eval.parent(substitute(optimizer[["SQN"]] <- SQN_obj))
		eval.parent(substitute(optimizer[["initialized"]]  <- as.logical(TRUE)))
		eval.parent(substitute(optimizer[["saved_params"]] <- NULL))
		grad_init <- vector(mode = "numeric", length =  SQN_obj$n)
		eval.parent(substitute(optimizer[["gradient"]] <- grad_init))
		if (!SQN_obj$use_grad_diff) {
			hess_vec_init <- vector(mode = "numeric", length =  SQN_obj$n)
		} else {
			hess_vec_init <- vector(mode = "numeric", length =  1)
		}
		eval.parent(substitute(optimizer[["hess_vec"]] <- hess_vec_init))
		optimizer$SQN      <- SQN_obj
		optimizer$gradient <- grad_init
		optimizer$hess_vec <- hess_vec_init
	}
	if (NROW(x) != optimizer$SQN$n) stop("'x' has wrong dimensions.")
	
	req_vec <- vector(mode = "numeric", length =  optimizer$SQN$n)
	if (!optimizer$SQN$use_grad_diff) {
		req_hess_vec <- vector(mode = "numeric", length =  optimizer$SQN$n)
	} else {
		req_hess_vec <- vector(mode = "numeric", length =  1)
	}
	x_changed <- as.integer(0)
	iter_info <- as.integer(0)
	task      <- as.integer(0)
	
	.Call("r_run_SQN", optimizer$SQN$BFGS_mem$s_mem, optimizer$SQN$BFGS_mem$y_mem,
		  optimizer$SQN$BFGS_mem$buffer_rho, optimizer$SQN$BFGS_mem$buffer_alpha,
		  optimizer$SQN$BFGS_mem$s_bak, optimizer$SQN$BFGS_mem$y_bak,
		  optimizer$SQN$BFGS_mem$mem_size, optimizer$SQN$BFGS_mem$mem_used,
		  optimizer$SQN$BFGS_mem$mem_st_ix, optimizer$SQN$BFGS_mem$upd_freq,
		  optimizer$SQN$BFGS_mem$y_reg, optimizer$SQN$BFGS_mem$min_curvature,
		  optimizer$SQN$grad_prev, optimizer$SQN$x_sum,
		  optimizer$SQN$x_avg_prev, optimizer$SQN$use_grad_diff,
		  optimizer$SQN$niter, optimizer$SQN$section, optimizer$SQN$nthreads,
		  optimizer$SQN$check_nan, optimizer$SQN$n,
		  x, optimizer$gradient, optimizer$hess_vec, step_size,
		  x_changed, req_vec,req_hess_vec, task, iter_info)
	
	if (get.task(task) != "calc_hess_vec") req_hess_vec <- NULL
	return(list(
		task = get.task(task),
		requested_on = list(req_x = req_vec, req_vec = req_hess_vec),
		info = list(
			x_changed_in_run = get.x.changed(x_changed),
			iteration_number = as.integer(optimizer$SQN$niter) + 1 - 1,
			iteration_info   = get.iter.info(iter_info)
			)
	))
}

#' @title Run adaQN optimizer in free-mode
#' @description Run the next step of an adaQN optimization procedure, after the last requested calculation
#' has been fed to the optimizer. When run for the first time, there is no request, so the function just
#' needs to be run on the object as it is returned from function `adaQN_free`.
#' @param optimizer An `adaQN_free` optimizer, for which its last request must have been served. Will be updated in-place.
#' @param x Current values of the variables being optimized. Must be a numeric vector. Will be updated in-place.
#' @param step_size Step size for the quasi-Newton update.
#' @return A request with the next piece of required information. The output will be a list with the following levels:
#' \itemize{
#'    \item{task} Requested task (one of "calc_grad", "calc_fun_val_batch", "calc_grad_big_batch").
#'    \item{requested_on} Values of `x` at which the requested information must be calculated.
#'    \item{info} \itemize{
#'       \item{x_changed_in_run} Whether the `x` vector was updated.
#'       \item{iteration_number} Current iteration number (in terms of quasi-Newton updates).
#'       \item{iteration_info} Information about potential problems encountered during the iteration.
#'    } 
#' }
#' @export
#' @seealso \link{adaQN_free}
run_adaQN_free <- function(optimizer, x, step_size) {
	check.x.and.step.size(x, step_size)
	if (!("adaQN_free" %in% class(optimizer))) stop("This function only applies to free-mode adaQN optimizer.")
	
	if (!optimizer$initialized) {
		adaQN_obj <- create.r.adaQN(NROW(x), optimizer$saved_params$mem_size, optimizer$saved_params$fisher_size,
									optimizer$saved_params$bfgs_upd_freq, optimizer$saved_params$max_incr,
									optimizer$saved_params$min_curvature, optimizer$saved_params$scal_reg,
									optimizer$saved_params$rmsprop_weight, optimizer$saved_params$use_grad_diff,
									optimizer$saved_params$y_reg, optimizer$saved_params$check_nan,
									optimizer$saved_params$nthreads)
		eval.parent(substitute(optimizer[["adaQN"]] <- adaQN_obj))
		eval.parent(substitute(optimizer[["initialized"]]  <- as.logical(TRUE)))
		eval.parent(substitute(optimizer[["saved_params"]] <- NULL))
		grad_init <- vector(mode = "numeric", length =  adaQN_obj$n)
		eval.parent(substitute(optimizer[["gradient"]] <- grad_init))
		eval.parent(substitute(optimizer[["f"]] <- as.numeric(0)))
		optimizer$adaQN    <- adaQN_obj
		optimizer$gradient <- grad_init
		optimizer$f        <- as.numeric(0)
	}
	if (NROW(x) != optimizer$adaQN$n) stop("'x' has wrong dimensions.")
	
	req_vec   <- vector(mode = "numeric", length =  optimizer$adaQN$n)
	x_changed <- as.integer(0)
	iter_info <- as.integer(0)
	task      <- as.integer(0)
	
	.Call("r_run_adaQN", optimizer$adaQN$BFGS_mem$s_mem, optimizer$adaQN$BFGS_mem$y_mem,
		  optimizer$adaQN$BFGS_mem$buffer_rho, optimizer$adaQN$BFGS_mem$buffer_alpha,
		  optimizer$adaQN$BFGS_mem$s_bak, optimizer$adaQN$BFGS_mem$y_bak,
		  optimizer$adaQN$BFGS_mem$mem_size, optimizer$adaQN$BFGS_mem$mem_used,
		  optimizer$adaQN$BFGS_mem$mem_st_ix, optimizer$adaQN$BFGS_mem$upd_freq,
		  optimizer$adaQN$BFGS_mem$y_reg, optimizer$adaQN$BFGS_mem$min_curvature,
		  optimizer$adaQN$Fisher_mem$`F`, optimizer$adaQN$Fisher_mem$buffer_y,
		  optimizer$adaQN$Fisher_mem$mem_size, optimizer$adaQN$Fisher_mem$mem_used,
		  optimizer$adaQN$Fisher_mem$mem_st_ix,
		  optimizer$adaQN$H0, optimizer$adaQN$grad_prev, optimizer$adaQN$x_sum, optimizer$adaQN$x_avg_prev,
		  optimizer$adaQN$grad_sum_sq, optimizer$adaQN$f_prev, optimizer$adaQN$max_incr, optimizer$adaQN$scal_reg,
		  optimizer$adaQN$rmsprop_weight, optimizer$adaQN$use_grad_diff,
		  optimizer$adaQN$niter, optimizer$adaQN$section, optimizer$adaQN$nthreads,
		  optimizer$adaQN$check_nan, optimizer$adaQN$n,
		  x, optimizer$f, optimizer$gradient, step_size,
		  x_changed, req_vec, task, iter_info)
	return(list(
		task = get.task(task),
		requested_on = req_vec,
		info = list(
			x_changed_in_run = get.x.changed(x_changed),
			iteration_number = as.integer(optimizer$adaQN$niter) + 1 - 1,
			iteration_info   = get.iter.info(iter_info)
		)
	))
}

#' @title Update objective function value (adaQN)
#' @description Update the (expected) value of the objective function in an `adaQN_free` object, after
#' it has been requested by the optimizer (do NOT update it otherwise).
#' @param optimizer An `adaQN_free` object which after the last run had requested a new function evaluation.
#' @param fun Function as evaluated (in expectation) on the values of `x` that were returned in the request.
#' @return No return value (object is updated in-place).
#' @export
update_fun <- function(optimizer, fun) {
	if (!("adaQN_free" %in% class(optimizer))) {
		stop("'update_fun' is only applicable for adaQN optimizer.")
	}
	if (is.null(fun)) stop("'fun' cannot be missing.")
	if ("integer" %in% class(fun)) fun <- as.numeric(fun)
	if (!("numeric" %in% class(fun))) stop("'fun' must be a number.")
	if (NROW(fun) > 1) stop("'fun' must be a single number.")
	.Call("copy_vec", fun, optimizer$f, as.integer(1))
}

#' @title Update gradient (oLBFGS, SQN, adaQN)
#' @description Update the (expected) gradient in an optimizer from this package, after
#' it has been requested by the optimizer (do NOT update it otherwise).
#' @param optimizer A free-mode optimizer from this package (`oLBFGS_free`, `SQN_free`, `adaQN_free`) which
#' after the last run had requested a new gradient evaluation..
#' @param gradient The (expected value of the) gradient as evaluated on the values of `x` that
#' were returned in the request. Must be a numeric vector.
#' @return No return value (object is updated in-place).
#' @export
update_gradient <- function(optimizer, gradient) {
	if (!NROW(intersect(class(optimizer), c("oLBFGS_free", "SQN_free", "adaQN_free")))) {
		stop("'optimizer' must be one of the free-mode optimizers from this package.")
	}
	if (is.null(gradient)) stop("'gradient' cannot be missing.")
	if ("integer" %in% class(gradient)) gradient <- as.numeric(gradient)
	if (!("numeric" %in% class(gradient))) stop("'gradient' must be a numeric vector")
	curr_n = optimizer$oLBFGS$n
	if (is.null(curr_n)) curr_n <- optimizer$SQN$n; if (is.null(curr_n)) curr_n <- optimizer$adaQN$n;
	if (NROW(gradient) != curr_n) stop("'gradient' must be of the same dimensionality as 'x'.")
	.Call("copy_vec", gradient, optimizer$gradient, curr_n)
}

#' @title Update Hessian-vector product (SQN)
#' @description Update the (expected) values of the Hessian-vector product in an `SQN_free` object,
#' after it has been requested by the optimizer (do NOT update it otherwise).
#' @param optimizer An `SQN_free` optimizer which after the last run had requested a new Hessian-vector evaluation.
#' @param hess_vec The (expected) value of the Hessian evaluated at the values of `x` that were returned in the
#' request, multiplied by the vector that was returned in the same request. Must be a numeric vector.
#' @return No return value (object is updated in-place).
#' @export
update_hess_vec <- function(optimizer, hess_vec) {
	if (!("SQN_free" %in% class(optimizer))) {
		stop("'update_hess_vec' is only applicable for SQN optimizer.")
	}
	if (is.null(hess_vec)) stop("'hess_vec' cannot be missing.")
	if ("integer" %in% class(hess_vec)) hess_vec <- as.numeric(hess_vec)
	if (!("numeric" %in% class(hess_vec))) stop("'hess_vec' must be a numeric vector")
	if (NROW(hess_vec) != optimizer$SQN$n) stop("'hess_vec' must be of the same dimensionality as 'x'.")
	.Call("copy_vec", hess_vec, optimizer$hess_vec, optimizer$SQN$n)
}

#' @title Print summary info about oLBFGS free-mode object
#' @param x An `oLBFGS_free` object as output by function of the same name.
#' @param ... Not used.
#' @export
print.oLBFGS_free <- function(x, ...) {
	cat("oLBFGS free-mode optimizer\n\n")
	if (!x$initialized) {
		cat("Optimizer has not yet been run.")
	} else {
		cat(sprintf("Optimizing function with %d variables\n", x$oLBFGS$n))
		cat(sprintf("Iteration number: %d\n", x$oLBFGS$niter))
	}
}

#' @title Print summary info about SQN free-mode object
#' @param x An `SQN_free` object as output by function of the same name.
#' @param ... Not used.
#' @export
print.SQN_free <- function(x, ...) {
	cat("SQN free-mode optimizer\n\n")
	if ((x$initialized && x$SQN$use_grad_diff) || (!x$initialized && x$saved_params$use_grad_diff)) {
		cat("Using gradient differencing\n")
	}
	if (!x$initialized) {
		cat("Optimizer has not yet been run.")
	} else {
		cat(sprintf("Optimizing function with %d variables\n", x$SQN$n))
		cat(sprintf("Iteration number: %d\n", x$SQN$niter))
		cat(sprintf("Current number of correction pairs: %d\n", x$SQN$BFGS_mem$mem_used))
	}
}

#' @title Print summary info about adaQN free-mode object
#' @param x An `adaQN_free` object as output by function of the same name.
#' @param ... Not used.
#' @export
print.adaQN_free <- function(x, ...) {
	cat("adaQN free-mode optimizer\n\n")
	if ((x$initialized && x$adaQN$use_grad_diff) || (!x$initialized && x$saved_params$use_grad_diff)) {
		cat("Using gradient differencing\n")
	}
	if (!x$initialized) {
		cat("Optimizer has not yet been run.")
	} else {
		cat(sprintf("Optimizing function with %d variables\n", x$adaQN$n))
		cat(sprintf("Iteration number: %d\n", x$adaQN$niter))
		cat(sprintf("Current number of correction pairs: %d\n", x$adaQN$BFGS_mem$mem_used))
		if (!x$adaQN$use_grad_diff) {
			cat(sprintf("Current size of Fisher memory: %d\n", x$adaQN$Fisher_mem$mem_used))
		}
	}
}
