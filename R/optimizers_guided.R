#' @title Predict function for stochastic optimizer object
#' @description Calls the user-defined predict function for an object
#' optimized through this package's functions.
#' @param object Optimizer from this module as output by functions `oLBFGS`, `SQN`, `adaQN`. Must
#' have been constructed with a predict function.
#' @param newdata Data on which to make predictions (will be passed to the user-provided function).
#' @param ... Additional arguments to pass to the user-provided predict function.
#' @seealso \link{oLBFGS} , \link{SQN} , \link{adaQN}
#' @export
predict.stochQN_guided <- function(object, newdata, ...) {
	if (is.null(object$pred_fun)) stop("Optimizer was constructed without a predict function.")
	return(object$pred_fun(newdata, object$x0, ...))
}

#' @title Partial fit stochastic model to new data
#' @description Runs one iteration of the stochastic optimizer on the new data passed here.
#' @param optimizer A stochastic optimizer from this package as output by functions `oLBFGS`, `SQN`, `adaQN`.
#' Will be modified in-place.
#' @param X Covariates to pass to the user-defined gradient / objective / Hessian-vector functions.
#' @param y Target variable to pass to the user-defined gradient / objective / Hessian-vector functions.
#' @param weights Target variable to pass to the user-defined gradient / objective / Hessian-vector functions.
#' @param ... Additional arguments to pass to the user-defined gradient / objective / Hessian-vector functions.
#' @return No return value (object is modified in-place).
#' @seealso \link{oLBFGS} , \link{SQN} , \link{adaQN}
#' @export
partial_fit <- function(optimizer, X, y = NULL, weights = NULL, ...) {
	this    <- optimizer
	new_req <- this$req
	
	if ("oLBFGS" %in% class(this)) {
		curr_iter <- this$optimizer$oLBFGS$niter + 1 - 1
	} else if ("SQN" %in% class(this)) {
		curr_iter <- this$optimizer$SQN$niter + 1 - 1
	} else if ("adaQN" %in% class(this)) {
		curr_iter <- this$optimizer$adaQN$niter + 1 - 1
	} else {
		stop("Invalid optimizer object.")
	}
	
	while (TRUE) {

		### Pass required calculation
		if (new_req$task == "calc_grad") {
			grad <- this$grad_fun(this$x0, X, y, weights, ...)
			update_gradient(this$optimizer, grad)
		} else if (new_req$task == "calc_grad_same_batch") {
			grad <- this$grad_fun(this$x0, X, y, weights, ...)
			update_gradient(this$optimizer, grad)
		} else if (new_req$task == "calc_grad_big_batch") {
			this <- get.saved.batch(this)
			grad <- this$grad_fun(this$x0, this$long_batch$X, this$long_batch$y, this$long_batch$w, ...)
			update_gradient(this$optimizer, grad)
			this$long_batch <- NULL
		} else if (new_req$task == "calc_hess_vec") {
			this <- get.saved.batch(this)
			hess_vec <- this$hess_vec_fun(this$x0, new_req$requested_on$req_vec,
										  this$long_batch$X, this$long_batch$y, this$long_batch$w,
										  ...)
			update_hess_vec(this$optimizer, hess_vec)
			this$long_batch <- NULL
		} else if (new_req$task == "calc_fun_val_batch") {
			this <- get.saved.batch(this)
			fun  <- this$obj_fun(this$x0, this$long_batch$X, this$long_batch$y, this$long_batch$w, ...)
			update_fun(this$optimizer, fun)
			this$long_batch <- NULL
		} else if (new_req$task == "invalid_input") {
			stop("Got invalid inputs from the functions")
		} else {
			stop("Unexpected error occurred. Optimization failed.")
		}
		
		### Run again
		if ("oLBFGS" %in% class(this)) {
			new_req <- run_oLBFGS_free(this$optimizer, this$x0, this$step_fun(this$optimizer$oLBFGS$niter) * this$initial_step)
		} else if ("SQN" %in% class(this)) {
			new_req <- run_SQN_free(this$optimizer, this$x0, this$step_fun(this$optimizer$SQN$niter) * this$initial_step)
		} else if ("adaQN" %in% class(this)) {
			new_req <- run_adaQN_free(this$optimizer, this$x0, this$step_fun(this$optimizer$adaQN$niter) * this$initial_step)
		}
		
		### Report potential problems if any were encountered
		if (this$verbose && new_req$info$iteration_info != "no_problems_encountered") {
			if ("oLBFGS" %in% class(this)) {
				opt_name <- "oLBFGS"
			} else if ("SQN" %in% class(this)) {
				opt_name <- "SQN"
			} else if ("adaQN" %in% class(this)) {
				opt_name <- "adaQN"
			}
			cat(sprintf("%s - at iteration %d: %s\n", opt_name, new_req$info$iteration_number, new_req$info$iteration_info))
		}

		### Stop if the iteration is over
		if (curr_iter < new_req$info$iteration_number) { break }
	}
	
	### Call the callback function if needed
	if (!is.null(this$callback_iter)) {
		this$callback_iter(this$x0, new_req$info$iteration_number, this$args_cb)
	}
	### Store this batch of data if needed
	this <- save.batch(this, X, y, weights)
	
	### Update the requested piece of info
	this$req <- new_req
	
	### Store back all the changes
	eval.parent(substitute(optimizer <- this))
	return(invisible(NULL))
}

#' @title oLBFGS guided optimizer
#' @description Optimizes an empirical (convex) loss function over batches of sample data.
#' @param x0 Initial values for the variables to optimize.
#' @param grad_fun Function taking as unnamed arguments `x_curr` (variable values), `X` (covariates),
#' `y` (target variable), and `w` (weights), plus additional arguments (`...`), and producing the expected
#' value of the gradient when evalauted on that data.
#' @param pred_fun Function taking an unnamed argument as data, another unnamed argument as the variable values,
#' and optional extra arguments (`...`). Will be called when using `predict` on the object returned by this function.
#' @param initial_step Initial step size.
#' @param step_fun Function accepting the iteration number as an unnamed parameter, which will output the
#' number by which `initial_step` will be multiplied at each iteration to get the step size for that
#' iteration.
#' @param callback_iter Callback function which will be called at the end of each iteration.
#' Will pass three unnamed arguments: the current variable values, the current iteration number,
#' and `args_cb`. Pass `NULL` if there is no need to call a callback function.
#' @param args_cb Extra argument to pass to the callback function.
#' @param verbose Whether to print information about iteration statuses when something goes wrong.
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
#' @return an `oLBFGS` object with the user-supplied functions, which can be fit to batches of data
#' through function `partial_fit`, and can produce predictions on new data through function `predict`.
#' @seealso \link{partial_fit} , \link{predict.stochQN_guided} , \link{oLBFGS_free}
#' @references \itemize{ \item Schraudolph, N.N., Yu, J. and Guenter, S., 2007, March.
#' "A stochastic quasi-Newton method for online convex optimization."
#' In Artificial Intelligence and Statistics (pp. 436-443).
#' \item Wright, S. and Nocedal, J., 1999. "Numerical optimization." (ch 7) Springer Science, 35(67-68), p.7.}
#' @examples
#' ### Example regression with randomly-generated data
#' library(stochQN)
#' 
#' ### Will sample data y ~ Ax + epsilon
#' true_coefs <- c(1.12, 5.34, -6.123)
#' 
#' generate_data_batch <- function(true_coefs, n = 100) {
#'   X <- matrix(
#'     rnorm(length(true_coefs) * n),
#'     nrow=n, ncol=length(true_coefs))
#'   y <- X %*% true_coefs + rnorm(n)
#'   return(list(X = X, y = y))
#' }
#' 
#' ### Regular regression function that minimizes RMSE
#' eval_fun <- function(coefs, X, y, weights=NULL, lambda=1e-5) {
#'   pred <- as.numeric(X %*% coefs)
#'   RMSE <- sqrt(mean((pred - y)^2))
#'   reg  <- lambda * as.numeric(coefs %*% coefs)
#'   return(RMSE + reg)
#' }
#' 
#' eval_grad <- function(coefs, X, y, weights=NULL, lambda=1e-5) {
#'   pred <- X %*% coefs
#'   grad <- colMeans(X * as.numeric(pred - y))
#'   grad <- grad + 2 * lambda * as.numeric(coefs^2)
#'   return(grad)
#' }
#' 
#' pred_fun <- function(X, coefs, ...) {
#'   return(as.numeric(X %*% coefs))
#' }
#' 
#' ### Initialize optimizer form arbitrary values
#' x0 <- c(1, 1, 1)
#' optimizer <- oLBFGS(x0, grad_fun=eval_grad,
#'   pred_fun=pred_fun, initial_step=1e-1)
#' val_data <- generate_data_batch(true_coefs, n=100)
#' 
#' ### Fit to 50 batches of data, 100 observations each
#' set.seed(1)
#' for (i in 1:50) {
#'   new_batch <- generate_data_batch(true_coefs, n=100)
#'   partial_fit(
#'     optimizer,
#'     new_batch$X, new_batch$y,
#'     lambda=1e-5)
#'   x_curr <- get_curr_x(optimizer)
#'   i_curr <- get_iteration_number(optimizer)
#'   if ((i_curr %% 10)  == 0) {
#'     cat(sprintf(
#'       "Iteration %d - E[f(x)]: %f - values of x: [%f, %f, %f]\n",
#'       i_curr,
#'       eval_fun(x_curr, val_data$X, val_data$y, lambda=1e-5),
#'       x_curr[1], x_curr[2], x_curr[3]))
#'   }
#' }
#' 
#' ### Predict for new data
#' new_batch <- generate_data_batch(true_coefs, n=10)
#' yhat <- predict(optimizer, new_batch$X)
#' @export
oLBFGS <- function(x0, grad_fun, pred_fun = NULL,
				   initial_step = 1e-2, step_fun = function(iter) 1/sqrt((iter/10)+1),
				   callback_iter = NULL, args_cb = NULL, verbose = TRUE,
				   mem_size = 10, hess_init = NULL, min_curvature = 1e-4, y_reg = NULL,
				   check_nan = TRUE, nthreads = -1) {
	this <- list()
	this <- take.attributes.guided(this, x0, initial_step, step_fun, NULL, grad_fun, NULL,
								   pred_fun, callback_iter, args_cb, verbose)
	this$optimizer <- oLBFGS_free(mem_size, hess_init, min_curvature, y_reg, check_nan, nthreads)
	this$req       <- run_oLBFGS_free(this$optimizer, this$x0, this$initial_step)
	this$prev_iter <- 0
	
	class(this) <- c("oLBFGS", "stochQN_guided")
	return(this)
}

#' @title Print summary info about oLBFGS guided-mode object
#' @param x An `oLBFGS` object as output by function of the same name.
#' @param ... Not used.
#' @export
print.oLBFGS <- function(x, ...) {
	cat("oLBFGS optimizer\n\n")
	cat(sprintf("Optimizing function with %d variables\n", x$optimizer$oLBFGS$n))
	cat(sprintf("Iteration number: %d\n", x$optimizer$oLBFGS$niter))
}

#' @title SQN guided optimizer
#' @description Optimizes an empirical (convex) loss function over batches of sample data.
#' @param x0 Initial values for the variables to optimize.
#' @param grad_fun Function taking as unnamed arguments `x_curr` (variable values), `X` (covariates),
#' `y` (target variable), and `w` (weights), plus additional arguments (`...`), and producing the expected
#' value of the gradient when evalauted on that data.
#' @param hess_vec_fun Function taking as unnamed arguments `x_curr` (variable values), `vec` (numeric vector),
#' `X` (covariates), `y` (target variable), and `w` (weights), plus additional arguments (`...`), and producing
#' the expected value of the Hessian (with variable values at `x_curr`) when evalauted on that data, multiplied
#' by the vector `vec`. Not required when using `use_grad_diff` = `TRUE`.
#' @param pred_fun Function taking an unnamed argument as data, another unnamed argument as the variable values,
#' and optional extra arguments (`...`). Will be called when using `predict` on the object returned by this function.
#' @param initial_step Initial step size.
#' @param step_fun Function accepting the iteration number as an unnamed parameter, which will output the
#' number by which `initial_step` will be multiplied at each iteration to get the step size for that
#' iteration.
#' @param callback_iter Callback function which will be called at the end of each iteration.
#' Will pass three unnamed arguments: the current variable values, the current iteration number,
#' and `args_cb`. Pass `NULL` if there is no need to call a callback function.
#' @param args_cb Extra argument to pass to the callback function.
#' @param verbose Whether to print information about iteration statuses when something goes wrong.
#' @param mem_size Number of correction pairs to store for approximation of Hessian-vector products.
#' @param bfgs_upd_freq Number of iterations (batches) after which to generate a BFGS correction pair.
#' @param min_curvature Minimum value of (s * y) / (s * s) in order to accept a correction pair. Pass `NULL` for
#' no minimum.
#' @param y_reg Regularizer for 'y' vector (gets added y_reg * s). Pass `NULL` for no regularization.
#' @param use_grad_diff Whether to create the correction pairs using differences between gradients instead of Hessian-vector products.
#' These gradients are calculated on a larger batch than the regular ones (given by batch_size * bfgs_upd_freq).
#' @param check_nan Whether to check for variables becoming NaN after each iteration, and reverting the step if they do
#' (will also reset BFGS memory).
#' @param nthreads  Number of parallel threads to use. If set to -1, will determine the number of available threads and use
#' all of them. Note however that not all the computations can be parallelized.
#' @return an `SQN` object with the user-supplied functions, which can be fit to batches of data
#' through function `partial_fit`, and can produce predictions on new data through function `predict`.
#' @seealso \link{partial_fit} , \link{predict.stochQN_guided} , \link{SQN_free}
#' @references \itemize{ \item Byrd, R.H., Hansen, S.L., Nocedal, J. and Singer, Y., 2016.
#' "A stochastic quasi-Newton method for large-scale optimization."
#' SIAM Journal on Optimization, 26(2), pp.1008-1031.
#' \item Wright, S. and Nocedal, J., 1999. "Numerical optimization." (ch 7) Springer Science, 35(67-68), p.7.}
#' @examples 
#' ### Example logistic regression with randomly-generated data
#' library(stochQN)
#' 
#' ### Will sample data y ~ Bernoulli(sigm(Ax))
#' true_coefs <- c(1.12, 5.34, -6.123)
#' 
#' generate_data_batch <- function(true_coefs, n = 100) {
#'   X <- matrix(rnorm(length(true_coefs) * n), nrow=n, ncol=length(true_coefs))
#'   y <- 1 / (1 + exp(-as.numeric(X %*% true_coefs)))
#'   y <- as.numeric(y >= runif(n))
#'   return(list(X = X, y = y))
#' }
#' 
#' ### Logistic regression likelihood/loss
#' eval_fun <- function(coefs, X, y, weights=NULL, lambda=1e-5) {
#'   pred    <- 1 / (1 + exp(-as.numeric(X %*% coefs)))
#'   logloss <- mean(-(y * log(pred) + (1 - y) * log(1 - pred)))
#'   reg     <- lambda * as.numeric(coefs %*% coefs)
#'   return(logloss + reg)
#' }
#' 
#' eval_grad <- function(coefs, X, y, weights=NULL, lambda=1e-5) {
#'   pred <- 1 / (1 + exp(-(X %*% coefs)))
#'   grad <- colMeans(X * as.numeric(pred - y))
#'   grad <- grad + 2 * lambda * as.numeric(coefs^2)
#'   return(as.numeric(grad))
#' }
#' 
#' eval_Hess_vec <- function(coefs, vec, X, y, weights=NULL, lambda=1e-5) {
#'   pred <- 1 / (1 + exp(-as.numeric(X %*% coefs)))
#'   diag <- pred * (1 - pred)
#'   Hp   <- (t(X) * diag) %*% (X %*% vec)
#'   Hp   <- Hp / NROW(X) + 2 * lambda * vec
#'   return(as.numeric(Hp))
#' }
#' 
#' pred_fun <- function(X, coefs, ...) {
#'   return(1 / (1 + exp(-as.numeric(X %*% coefs))))
#' }
#' 
#' 
#' ### Initialize optimizer form arbitrary values
#' x0 <- c(1, 1, 1)
#' optimizer <- SQN(x0, grad_fun=eval_grad, pred_fun=pred_fun,
#'   hess_vec_fun=eval_Hess_vec, initial_step=1e-0)
#' val_data <- generate_data_batch(true_coefs, n=100)
#' 
#' ### Fit to 250 batches of data, 100 observations each
#' set.seed(1)
#' for (i in 1:250) {
#'   new_batch <- generate_data_batch(true_coefs, n=100)
#'   partial_fit(optimizer, new_batch$X, new_batch$y, lambda=1e-5)
#'   x_curr <- get_curr_x(optimizer)
#'   i_curr <- get_iteration_number(optimizer)
#'   if ((i_curr %% 10)  == 0) {
#'     cat(sprintf("Iteration %3d - E[f(x)]: %f - values of x: [%f, %f, %f]\n",
#'       i_curr, eval_fun(x_curr, val_data$X, val_data$y, lambda=1e-5),
#'       x_curr[1], x_curr[2], x_curr[3]))
#'   }
#' }
#' 
#' ### Predict for new data
#' new_batch <- generate_data_batch(true_coefs, n=10)
#' yhat <- predict(optimizer, new_batch$X)
#' @export
SQN <- function(x0, grad_fun, hess_vec_fun = NULL, pred_fun = NULL,
				initial_step = 1e-3, step_fun = function(iter) 1/sqrt((iter/10)+1),
				callback_iter = NULL, args_cb = NULL, verbose = TRUE,
				mem_size = 10, bfgs_upd_freq = 20, min_curvature = 1e-4, y_reg = NULL,
				use_grad_diff = FALSE, check_nan = TRUE, nthreads = -1) {
	this <- list()
	this <- take.attributes.guided(this, x0, initial_step, step_fun, NULL, grad_fun, hess_vec_fun,
								   pred_fun, callback_iter, args_cb, verbose)
	this$optimizer <- SQN_free(mem_size, bfgs_upd_freq, min_curvature, y_reg,
							   use_grad_diff, check_nan, nthreads)
	this$req       <- run_SQN_free(this$optimizer, this$x0, this$initial_step)
	this$prev_iter <- 0
	
	if ((!this$optimizer$SQN$use_grad_diff) & is.null(hess_vec_fun)) {
		stop("Must pass Hessian-vector function when not using 'use_grad_diff'.")
	}
	
	class(this) <- c("SQN", "stochQN_guided")
	return(this)
}

#' @title Print summary info about SQN guided-mode object
#' @param x An `SQN` object as output by function of the same name.
#' @param ... Not used.
#' @export
print.SQN <- function(x, ...) {
	cat("SQN optimizer\n\n")
	cat(sprintf("Optimizing function with %d variables\n", x$optimizer$SQN$n))
	cat(sprintf("Iteration number: %d\n", x$optimizer$SQN$niter))
	if (!x$optimizer$SQN$use_grad_dif) {
		cat("Using Hessian-vector products.\n")
	} else {
		cat("Using gradient differencing.\n")
	}
}

#' @title adaQN guided optimizer
#' @description Optimizes an empirical (possibly non-convex) loss function over batches of sample data.
#' @param x0 Initial values for the variables to optimize.
#' @param grad_fun Function taking as unnamed arguments `x_curr` (variable values), `X` (covariates),
#' `y` (target variable), and `w` (weights), plus additional arguments (`...`), and producing the expected
#' value of the gradient when evalauted on that data.
#' @param obj_fun Function taking as unnamed arguments `x_curr` (variable values), `X` (covariates),
#' `y` (target variable), and `w` (weights), plus additional arguments (`...`), and producing the expected
#' value of the objective function when evalauted on that data. Only required when using `max_incr`.
#' @param pred_fun Function taking an unnamed argument as data, another unnamed argument as the variable values,
#' and optional extra arguments (`...`). Will be called when using `predict` on the object returned by this function.
#' @param initial_step Initial step size.
#' @param step_fun Function accepting the iteration number as an unnamed parameter, which will output the
#' number by which `initial_step` will be multiplied at each iteration to get the step size for that
#' iteration.
#' @param callback_iter Callback function which will be called at the end of each iteration.
#' Will pass three unnamed arguments: the current variable values, the current iteration number,
#' and `args_cb`. Pass `NULL` if there is no need to call a callback function.
#' @param args_cb Extra argument to pass to the callback function.
#' @param verbose Whether to print information about iteration statuses when something goes wrong.
#' @param mem_size Number of correction pairs to store for approximation of Hessian-vector products.
#' @param fisher_size Number of gradients to store for calculation of the empirical Fisher product with gradients.
#' If passing `NULL`, will force `use_grad_diff` to `TRUE`.
#' @param bfgs_upd_freq Number of iterations (batches) after which to generate a BFGS correction pair.
#' @param max_incr Maximum ratio of function values in the validation set under the average values of `x` during current epoch
#' vs. previous epoch. If the ratio is above this threshold, the BFGS and Fisher memories will be reset, and `x`
#' values reverted to their previous average.
#' If not using a validation set, will take a longer batch for function evaluations (same as used for gradients	
#' when using `use_grad_diff` = `TRUE`).
#' Pass `NULL` for no function-increase checking.
#' @param min_curvature Minimum value of (s * y) / (s * s) in order to accept a correction pair. Pass `NULL` for
#' no minimum.
#' @param y_reg Regularizer for 'y' vector (gets added y_reg * s). Pass `NULL` for no regularization.
#' @param scal_reg Regularization parameter to use in the denominator for AdaGrad and RMSProp scaling.
#' @param rmsprop_weight If not `NULL`, will use RMSProp formula instead of AdaGrad for approximated inverse-Hessian initialization.
#' @param use_grad_diff Whether to create the correction pairs using differences between gradients instead of empirical Fisher matrix.
#' These gradients are calculated on a larger batch than the regular ones (given by batch_size * bfgs_upd_freq).
#' @param check_nan Whether to check for variables becoming NaN after each iteration, and reverting the step if they do
#' (will also reset BFGS memory).
#' @param nthreads Number of parallel threads to use. If set to -1, will determine the number of available threads and use
#' all of them. Note however that not all the computations can be parallelized.
#' @param X_val Covariates to use as validation set (only used when passing `max_incr`). If not passed, will use
#' a larger batch of stored data, in the same way as for Hessian-vector products in SQN.
#' @param y_val Target variable for the covariates to use as validation set (only used when passing `max_incr`).
#' If not passed, will use a larger batch of stored data, in the same way as for Hessian-vector products in SQN.
#' @param w_val Sample weights for the covariates to use as validation set (only used when passing `max_incr`).
#' If not passed, will use a larger batch of stored data, in the same way as for Hessian-vector products in SQN.
#' @return an `adaQN` object with the user-supplied functions, which can be fit to batches of data
#' through function `partial_fit`, and can produce predictions on new data through function `predict`.
#' @seealso \link{partial_fit} , \link{predict.stochQN_guided} , \link{adaQN_free}
#' @references \itemize{ \item Keskar, N.S. and Berahas, A.S., 2016, September.
#' "adaQN: An Adaptive Quasi-Newton Algorithm for Training RNNs."
#' In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 1-16). Springer, Cham.
#' \item Wright, S. and Nocedal, J., 1999. "Numerical optimization." (ch 7) Springer Science, 35(67-68), p.7.}
#' @examples 
#' ### Example regression with randomly-generated data
#' library(stochQN)
#' 
#' ### Will sample data y ~ Ax + epsilon
#' true_coefs <- c(1.12, 5.34, -6.123)
#' 
#' generate_data_batch <- function(true_coefs, n = 100) {
#'   X <- matrix(
#'     rnorm(length(true_coefs) * n),
#'     nrow=n, ncol=length(true_coefs))
#'   y <- X %*% true_coefs + rnorm(n)
#'   return(list(X = X, y = y))
#' }
#' 
#' ### Regular regression function that minimizes RMSE
#' eval_fun <- function(coefs, X, y, weights=NULL, lambda=1e-5) {
#'   pred <- as.numeric(X %*% coefs)
#'   RMSE <- sqrt(mean((pred - y)^2))
#'   reg  <- 2 * lambda * as.numeric(coefs %*% coefs)
#'   return(RMSE + reg)
#' }
#' 
#' eval_grad <- function(coefs, X, y, weights=NULL, lambda=1e-5) {
#'   pred <- X %*% coefs
#'   grad <- colMeans(X * as.numeric(pred - y))
#'   grad <- grad + 2 * lambda * as.numeric(coefs^2)
#'   return(grad)
#' }
#' 
#' pred_fun <- function(X, coefs, ...) {
#'   return(as.numeric(X %*% coefs))
#' }
#' 
#' ### Initialize optimizer form arbitrary values
#' x0 <- c(1, 1, 1)
#' optimizer <- adaQN(x0, grad_fun=eval_grad,
#'   pred_fun=pred_fun, obj_fun=eval_fun, initial_step=1e-0)
#' val_data <- generate_data_batch(true_coefs, n=100)
#' 
#' ### Fit to 50 batches of data, 100 observations each
#' for (i in 1:50) {
#'   set.seed(i)
#'   new_batch <- generate_data_batch(true_coefs, n=100)
#'   partial_fit(
#'     optimizer,
#'     new_batch$X, new_batch$y,
#'     lambda=1e-5)
#'   x_curr <- get_curr_x(optimizer)
#'   i_curr <- get_iteration_number(optimizer)
#'   if ((i_curr %% 10)  == 0) {
#'     cat(sprintf(
#'       "Iteration %d - E[f(x)]: %f - values of x: [%f, %f, %f]\n",
#'       i_curr,
#'       eval_fun(x_curr, val_data$X, val_data$y, lambda=1e-5),
#'       x_curr[1], x_curr[2], x_curr[3]))
#'   }
#' }
#' 
#' ### Predict for new data
#' new_batch <- generate_data_batch(true_coefs, n=10)
#' yhat <- predict(optimizer, new_batch$X)
#' @export
adaQN <- function(x0, grad_fun, obj_fun = NULL, pred_fun = NULL,
				  initial_step = 1e-2, step_fun = function(iter) 1/sqrt((iter/100)+1),
				  callback_iter = NULL, args_cb = NULL, verbose = TRUE,
				  mem_size = 10, fisher_size = 100, bfgs_upd_freq = 20, max_incr = 1.01,
				  min_curvature = 1e-4, y_reg = NULL, scal_reg = 1e-4,
				  rmsprop_weight = 0.9, use_grad_diff = FALSE, check_nan = TRUE, nthreads = -1,
				  X_val = NULL, y_val = NULL, w_val = NULL) {
	this <- list()
	this <- take.attributes.guided(this, x0, initial_step, step_fun, obj_fun, grad_fun, NULL,
								   pred_fun, callback_iter, args_cb, verbose)
	this$optimizer <- adaQN_free(mem_size, fisher_size, bfgs_upd_freq, max_incr,
								 min_curvature, scal_reg, rmsprop_weight,
								 y_reg, use_grad_diff, check_nan, nthreads)
	this$req       <- run_adaQN_free(this$optimizer, this$x0, this$initial_step)
	this$prev_iter <- 0
	
	if (this$optimizer$adaQN$max_incr > 0 & is.null(obj_fun)) {
		stop("Must pass objective function when using 'max_incr'.")
	}
	
	if (!is.null(X_val) | !is.null(y_val) | !is.null(w_val)) {
		this$valset <- list(X = X_val, y = y_val, w = w_val)
	}
	
	class(this) <- c("adaQN", "stochQN_guided")
	return(this)
}

#' @title Print summary info about adaQN guided-mode object
#' @param x An `adaQN` object as output by function of the same name.
#' @param ... Not used.
#' @export
print.adaQN <- function(x, ...) {
	cat("adaQN optimizer\n\n")
	cat(sprintf("Optimizing function with %d variables\n", x$optimizer$adaQN$n))
	cat(sprintf("Iteration number: %d\n", x$optimizer$adaQN$niter))
	if (x$optimizer$adaQN$rmsprop_weight > 0) {
		cat("Using RMSProp approx. Hessian initializer.\n")
	} else {
		cat("Using AgaGrad approx. Hessian initializer.\n")
	}
	if (!x$optimizer$adaQN$use_grad_dif) {
		cat("Using empirical Fisher matrix.\n")
	} else {
		cat("Using gradient differencing.\n")
	}
}

#' @title Get current values of the optimization variables
#' @param optimizer An optimizer (guided-mode) from this module, as output by
#' functions `oLBFGS`, `SQN`, `adaQN`.
#' @return A numeric vector with the current values of the variables being optimized.
#' @seealso \link{oLBFGS} , \link{SQN} , \link{adaQN}
#' @export
get_curr_x <- function(optimizer) {
	if (!NROW(intersect(class(optimizer), c("oLBFGS", "SQN", "adaQN")))) {
		stop("Function is only applicable for guided-mode optimizers from this package.")
	}
	return(optimizer$x0 + 1 - 1)
}

#' @title Get current iteration number from the optimizer object
#' @param optimizer An optimizer (guided-mode) from this module, as output by
#' functions `oLBFGS`, `SQN`, `adaQN`.
#' @return The current iteration number.
#' @seealso \link{oLBFGS} , \link{SQN} , \link{adaQN}
#' @export
get_iteration_number <- function(optimizer) {
	if (!NROW(intersect(class(optimizer), c("oLBFGS", "SQN", "adaQN")))) {
		stop("Function is only applicable for guided-mode optimizers from this package.")
	}
	if ("oLBFGS" %in% class(optimizer)) {
		niter <- optimizer$optimizer$oLBFGS$niter
	} else if ("SQN" %in% class(optimizer)) {
		niter <- optimizer$optimizer$SQN$niter
	} else if ("adaQN" %in% class(optimizer)) {
		niter <- optimizer$optimizer$adaQN$niter
	}
	return(niter + 1 - 1)
}
