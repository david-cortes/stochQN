take.common.inputs <- function(this, mem_size, min_curvature, y_reg, check_nan, nthreads) {
	if (mem_size < 0) stop("'mem_size' must be non-negative.")
	if ("numeric" %in% class(mem_size)) mem_size <- as.integer(mem_size)
	if (!("integer" %in% class(mem_size))) stop("'mem_size' must be an integer.")
	
	if (is.null(min_curvature) || is.na(min_curvature)) {
		min_curvature <- as.numeric(0)
	} else {
		if (!("numeric" %in% class(min_curvature))) stop("'min_curvature' must be a decimal number.")
		if (min_curvature < 0) stop("'min_curvature' must be non-negative.")
	}
	
	if (is.null(y_reg) || is.na(y_reg)) {
		y_reg <- as.numeric(0)
	} else {
		if (!("numeric" %in% class(y_reg))) stop("'y_reg' must be a decimal number.")
		if (y_reg < 0) stop("'y_reg' must be non-negative.")
	}
	
	if (is.null(nthreads) || is.na(nthreads)) {
		nthreads <- parallel::detectCores()
	}
	if ("numeric" %in% class(nthreads)) nthreads <- as.integer(nthreads)
	if (!("integer" %in% class(nthreads))) stop("'nthreads' must be an integer.")
	if (nthreads < 1) nthreads <- parallel::detectCores()
	
	this$mem_size      <- mem_size
	this$min_curvature <- min_curvature
	this$y_reg         <- y_reg
	this$check_nan     <- check.is.bool(check_nan, "check_nan")
	this$nthreads      <- nthreads
	return(this)
}

get.task <- function(task) {
	return(switch (as.character(task),
				   "100" = "invalid_input",
				   "101" = "calc_grad",
				   "102" = "calc_grad_same_batch",
				   "103" = "calc_grad_big_batch",
				   "104" = "calc_hess_vec",
				   "105" = "calc_fun_val_batch",
	))
}

get.iter.info <- function(iter_info) {
	return(switch(as.character(iter_info),
				  "200" = "no_problems_encountered",
				  "201" = "func_increased",
				  "202" = "curvature_too_small",
				  "203" = "search_direction_was_nan",
	))
}

get.x.changed <- function(x_changed) {
	return(switch(as.character(x_changed),
				  "0" = "did_not_update_x",
				  "1" = "updated_x",
				  "-1000" = "received_invalid_input"
	))
}

get.zero.as.int <- function() {
	### Note: for some reason, saving a list element as
	### lst$el <- as.integer(0)
	### will in some cases end up storing a different number
	### (probably some R bug)
	### The examples were failing because of this,
	### hence this function
	out <- vector(mode = "integer", length = 1)
	out[1] <- 0L
	return(out)
}

check.x.and.step.size <- function(x, step_size) {
	if (is.null(x) || any(is.na(x))) stop("'x' cannot be missing.")
	if (is.null(step_size) || is.na(step_size)) stop("'step_size' cannot be missing.")
	if (!("numeric" %in% class(x))) stop("'x' must be a numeric vector.")
	if (!("numeric" %in% class(step_size))) stop("'step_size' must be a decimal number.")
	if (NROW(step_size) > 1) stop("'step_size' must be a single number.")
}

check.positive.integer <- function(param, param_name) {
	if (is.null(param)) stop(paste0("'", param_name, "' cannot be missing."))
	if (NROW(param) > 1) stop(paste0("'", param_name, "' must be a single number."))
	if (is.na(param)) stop(paste0("'", param_name, "' cannot be NA."))
	if ("numeric" %in% class(param)) param <- as.integer(param)
	if (!("integer" %in% class(param))) stop(paste0("'", param_name, "' must be an integer."))
	if (param <= 0) stop(paste0("'", param_name, "' must be positive."))
	return(as.integer(param))
}

check.positive.float <- function(param, param_name) {
	if (NROW(param) > 1) stop(paste0("'", param_name, "' must be a single number."))
	if (is.na(param)) stop(paste0("'", param_name, "' cannot be NA."))
	if ("integer" %in% class(param)) param <- as.numeric(param)
	if (!("numeric" %in% class(param))) stop(paste0("'", param_name, "' must be an decimal number"))
	if (param <= 0) stop(paste0("'", param_name, "' must be positive."))
	return(as.numeric(param))
}

check.is.func <- function(fun, fun_name) {
	if (!is.null(fun)) {
		if (!("function" %in% class(fun))) {
			stop(paste0("'", fun_name, "' must be a function."))
		}
	}
}

check.is.bool <- function(var, var_name) {
	if (NROW(var) == 0 || NROW(var) > 1) stop(paste0("'", var_name, "' must be a single TRUE/FALSE."))
	return(as.logical(var))
}

take.attributes.guided <- function(this, x0, initial_step, step_fun, obj_fun, grad_fun, hess_vec_fun,
								   pred_fun, callback_iter, args_cb, verbose) {
	if (is.null(x0)) stop("'x0' cannot be missing.")
	if ("integer" %in% class(x0)) x0 <- as.numeric(x0)
	if (!("numeric" %in% class(x0))) stop("'x0' must be a numeric vector.")
	initial_step <- check.positive.float(initial_step, "initial_step")
	if (is.null(grad_fun)) stop("Gradient function cannot be missing.")
	if (is.null(step_fun)) stop("Step size function cannot be missing.")
	check.is.func(step_fun, "step_fun")
	check.is.func(obj_fun, "obj_fun")
	check.is.func(obj_fun, "obj_fun")
	check.is.func(grad_fun, "grad_fun")
	check.is.func(hess_vec_fun, "hess_vec_fun")
	check.is.func(pred_fun, "pred_fun")
	check.is.func(callback_iter, "callback_iter")
	verbose <- check.is.bool(verbose)
	
	
	this$x0            <- x0
	this$initial_step  <- initial_step
	this$step_fun      <- step_fun
	this$obj_fun       <- obj_fun
	this$grad_fun      <- grad_fun
	this$hess_vec_fun  <- hess_vec_fun
	this$pred_fun      <- pred_fun
	this$callback_iter <- callback_iter
	this$args_cb       <- args_cb
	this$verbose       <- verbose
	return(this)
}

reset.saved.batch <- function(this) {
	this$stored_samples_X <- list()
	this$stored_samples_y <- list()
	this$stored_samples_w <- list()
	return(this)
}

save.batch <- function(this, X, y, w) {
	if (!is.null(this$valset)) return(this)
	if (!is.null(X)) this$stored_samples_X[[length(this$stored_samples_X) + 1]] <- X
	if (!is.null(y)) this$stored_samples_y[[length(this$stored_samples_y) + 1]] <- y
	if (!is.null(w)) this$stored_samples_w[[length(this$stored_samples_w) + 1]] <- w
	if (
		(NROW(this$stored_samples_X) && NROW(this$stored_samples_y) && (NROW(this$stored_samples_X) != NROW(this$stored_samples_y))) ||
		(NROW(this$stored_samples_w) && NROW(this$stored_samples_y) && (NROW(this$stored_samples_w) != NROW(this$stored_samples_y))) ||
		(NROW(this$stored_samples_X) && NROW(this$stored_samples_w) && (NROW(this$stored_samples_X) != NROW(this$stored_samples_w)))
	) {
		stop("Cannot pass parameters 'X', 'y', 'w' only some times and not other times.")
	}
	return(this)
}

get.saved.batch <- function(this) {
	this$long_batch <- list()
	if (!is.null(this$valset)) {
		this$long_batch <- list(X = this$valset$X, y = this$valset$y, w = this$valset$w)
		return(this)
	}
	if (NROW(this$stored_samples_X)) {
		this$stored_samples_X$stringsAsFactors <- FALSE
		this$stored_samples_X$make.row.names <- FALSE
		this$long_batch$X <- do.call(rbind, this$stored_samples_X)
	}
	if (NROW(this$stored_samples_y)) {
		this$stored_samples_y$stringsAsFactors <- FALSE
		this$stored_samples_y$make.row.names <- FALSE
		this$long_batch$y <- do.call(rbind, this$stored_samples_y)
	}
	if (NROW(this$stored_samples_w)) {
		this$stored_samples_w$stringsAsFactors <- FALSE
		this$stored_samples_w$make.row.names <- FALSE
		this$long_batch$w <- do.call(rbind, this$stored_samples_w)
	}
	this <- reset.saved.batch(this)
	return(this)
}
