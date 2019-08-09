create.r.BFGS.mem <- function(mem_size, n, min_curvature, y_reg, upd_freq) {
	out       <- list()
	out$s_mem <- vector(mode = "numeric", length =  n * mem_size)
	out$y_mem <- vector(mode = "numeric", length =  n * mem_size)
	out$buffer_rho   <- vector(mode = "numeric", length =  mem_size)
	out$buffer_alpha <- vector(mode = "numeric", length =  mem_size)
	if (min_curvature > 0) {
		out$s_bak <- vector(mode = "numeric", length =  n)
		out$y_bak <- vector(mode = "numeric", length =  n)
	} else {
		out$s_bak <- vector(mode = "numeric", length =  1)
		out$y_bak <- vector(mode = "numeric", length =  1)
	}
	out$mem_size  <- as.integer(mem_size)
	out$mem_used  <- get.zero.as.int()
	out$mem_st_ix <- get.zero.as.int()
	out$upd_freq  <- as.integer(upd_freq)
	out$y_reg     <- as.numeric(y_reg)
	out$min_curvature <- as.numeric(min_curvature)
	return(out)
}

create.r.Fisher.mem <- function(mem_size, n) {
	out     <- list()
	out$`F` <- vector(mode = "numeric", length =  n * mem_size)
	out$buffer_y  <- vector(mode = "numeric", length =  mem_size)
	out$mem_size  <- as.integer(mem_size)
	out$mem_used  <- get.zero.as.int()
	out$mem_st_ix <- get.zero.as.int()
	return(out)
}

create.r.oLBFGS <- function(n, mem_size, hess_init, y_reg, min_curvature, check_nan, nthreads) {
	out <- list()
	out$BFGS_mem  <- create.r.BFGS.mem(mem_size, n, min_curvature, y_reg, 1)
	out$grad_prev <- vector(mode = "numeric", length =  n)
	out$hess_init <- as.numeric(hess_init)
	out$niter     <- get.zero.as.int()
	out$section   <- get.zero.as.int()
	out$nthreads  <- as.integer(nthreads)
	out$check_nan <- as.integer(as.logical(check_nan))
	out$n         <- as.integer(n)
	return(out)
}

create.r.SQN <- function(n, mem_size, bfgs_upd_freq, min_curvature,
						 use_grad_diff, y_reg, check_nan, nthreads) {
	out <- list()
	out$BFGS_mem <- create.r.BFGS.mem(mem_size, n, min_curvature, y_reg, bfgs_upd_freq)
	if (use_grad_diff) {
		out$grad_prev <- vector(mode = "numeric", length =  n)
	} else {
		out$grad_prev <- vector(mode = "numeric", length =  1)
	}
	out$x_sum         <- numeric(n)
	out$x_avg_prev    <- vector(mode = "numeric", length =  n)
	out$use_grad_diff <- as.integer(as.logical(use_grad_diff))
	out$niter     <- get.zero.as.int()
	out$section   <- get.zero.as.int()
	out$nthreads  <- as.integer(nthreads)
	out$check_nan <- as.integer(as.logical(check_nan))
	out$n         <- as.integer(n)
	return(out)
}

create.r.adaQN <- function(n, mem_size, fisher_size, bfgs_upd_freq,
						   max_incr, min_curvature, scal_reg, rmsprop_weight,
						   use_grad_diff, y_reg, check_nan, nthreads) {
	out <- list()
	out$BFGS_mem <- create.r.BFGS.mem(mem_size, n, min_curvature, y_reg, bfgs_upd_freq)
	if (use_grad_diff) {
		out$Fisher_mem <- create.r.Fisher.mem(1, 1)
		out$grad_prev  <- vector(mode = "numeric", length =  n)
	} else {
		out$Fisher_mem <- create.r.Fisher.mem(fisher_size, n)
		out$grad_prev  <- vector(mode = "numeric", length =  1)
	}
	out$H0          <- vector(mode = "numeric", length =  n)
	out$x_sum       <- numeric(n)
	out$x_avg_prev  <- vector(mode = "numeric", length =  n)
	out$grad_sum_sq <- numeric(n)
	
	out$max_incr       <- as.numeric(max_incr)
	out$scal_reg       <- as.numeric(scal_reg)
	out$rmsprop_weight <- as.numeric(rmsprop_weight)
	out$use_grad_diff  <- as.integer(use_grad_diff)
	out$f_prev    <- vector(mode = "numeric", length =  1)
	out$niter     <- get.zero.as.int()
	out$section   <- get.zero.as.int()
	out$nthreads  <- as.integer(nthreads)
	out$check_nan <- as.integer(as.logical(check_nan))
	out$n         <- as.integer(n)
	return(out)
}
