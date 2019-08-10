logistic_loss <- function(coefs, X, y, weights = NULL, lambda = 1e-5) {
	pred    <- 1 / (1 + exp(-as.numeric(X %*% coefs)))
	if (is.null(weights)) {
		logloss <- mean(-(y * log(pred) + (1 - y) * log(1 - pred)))
	} else {
		logloss <- sum( -(y * log(pred) + (1 - y) * log(1 - pred)) * weights) / sum(weights)
	}
	reg     <- lambda * as.numeric(coefs %*% coefs)
	return(logloss + reg)
}

logistic_grad <- function(coefs, X, y, weights = NULL, lambda = 1e-5) {
	pred <- 1 / (1 + exp(-(X %*% coefs)))
	if (is.null(weights)) {
		grad <- colMeans(X * as.numeric(pred - y))
	} else {
		grad <- colSums( X * as.numeric(pred - y)  * weights) / sum(weights)
	}
	grad <- grad + 2 * lambda * as.numeric(coefs)
	return(as.numeric(grad))
}

logistic_Hess_vec <- function(coefs, vec, X, y, weights = NULL, lambda = 1e-5) {
	pred <- 1 / (1 + exp(-as.numeric(X %*% coefs)))
	if (is.null(weights)) {
		diag <- pred * (1 - pred)
	} else {
		diag <- pred * (1 - pred) * weights
	}
	Hp   <- (t(X) * diag) %*% (X %*% vec)
	if (is.null(weights)) {
		Hp   <- Hp / NROW(X)      + 2 * lambda * vec
	} else {
		Hp   <- Hp / sum(weights) + 2 * lambda * vec
	}
	return(as.numeric(Hp))
}

logistic_pred <- function(X, coefs, ...) {
	return(1 / (1 + exp(-as.numeric(X %*% coefs))))
}

#' @title Stochastic Logistic Regression
#' @details Binary logistic regression, fit in batches using this package's own optimizers.
#' @param formula Formula for the model, if it is fit to data.frames instead of matrices/vectors.
#' @param pos_class If fit to data in the form of data.frames, a string indicating which of
#' the classes is the positive one. If fit to data in the form of matrices/vector, pass `NULL`.
#' @param dim Dimensionality of the model (number of features). Ignored when passing `formula` or when passing `x0`.
#' If the intercept is added from the option `intercept` here, it should not be counted towards `dim`.
#' @param intercept Whether to add an intercept to the covariates. Only ussed when fitting to matrices/vectors.
#' Ignored when passing formula (for formulas without intercept, put `-1` in the RHS to get rid of the intercept).
#' @param x0 Initial values of the variables. If passed, will ignore `dim` and `random_seed`. If not passed,
#' will generate random starting values ~ Norm(0, 0.1).
#' @param optimizer The optimizer to use - one of `adaQN` (recommended), `SQN`, `oLBFGS`.
#' @param optimizer_args Arguments to pass to the optimizer (same ones as the functions of the same name).
#' Must be a list. See the documentation of each optimizer for the parameters they take.
#' @param lambda Regularization parameter. Be aware that the functions assume the log-likelihood (a.k.a. loss)
#' is divided by the number of observations, so this number should be small.
#' @param random_seed Random seed to use for the initialization of the variables. Ignored when passing `x0`.
#' @param val_data Validation data (only used for `adaQN`). If passed, must be a list with entries `X`,
#' `y` (if passing data.frames for fitting), and optionally `w` (sample weights).
#' @return An object of class `stoch_logistic`, which can be fit to batches of data through functon `partial_fit_logistic`.
#' @seealso \link{partial_fit_logistic}, \link{coef.stoch_logistic} , \link{predict.stoch_logistic} , 
#' \link{adaQN} , \link{SQN}, \link{oLBFGS}
#' @examples 
#' library(stochQN)
#' 
#' ### Load Iris dataset
#' data("iris")
#' 
#' ### Example with X + y interface
#' X <- as.matrix(iris[, c("Sepal.Length", "Sepal.Width",
#'   "Petal.Length", "Petal.Width")])
#' y <- as.numeric(iris$Species == "setosa")
#' 
#' ### Initialize model with default parameters
#' model <- stochastic.logistic.regression(dim = 4)
#' 
#' ### Fit to 10 randomly-subsampled batches
#' batch_size <- as.integer(nrow(X) / 3)
#' for (i in 1:10) {
#'   set.seed(i)
#'   batch <- sample(nrow(X),
#'       size = batch_size, replace=TRUE)
#'   partial_fit_logistic(model, X, y)
#' }
#' 
#' ### Check classification accuracy
#' cat(sprintf(
#'   "Accuracy after 10 iterations: %.2f%%\n",
#'   100 * mean(
#'     predict(model, X, type = "class") == y)
#'   ))
#' 
#' 
#' ### Example with formula interface
#' iris_df <- iris
#' levels(iris_df$Species) <- c("setosa", "other", "other")
#' 
#' ### Initialize model with default parameters
#' model <- stochastic.logistic.regression(Species ~ .,
#'   pos_class="setosa")
#' 
#' ### Fit to 10 randomly-subsampled batches
#' batch_size <- as.integer(nrow(iris_df) / 3)
#' for (i in 1:10) {
#'   set.seed(i)
#'   batch <- sample(nrow(iris_df),
#'       size=batch_size, replace=TRUE)
#'   partial_fit_logistic(model, iris_df)
#' }
#' cat(sprintf(
#'   "Accuracy after 10 iterations: %.2f%%\n",
#'   100 * mean(
#'     predict(
#'       model, iris_df, type = "class") == iris_df$Species
#'       )
#'   ))
#' @export
stochastic.logistic.regression <- function(formula = NULL, pos_class = NULL, dim = NULL, intercept = TRUE, x0 = NULL,
										   optimizer = "adaQN", optimizer_args = list(initial_step = 1e-1, verbose = FALSE),
										   lambda = 1e-3, random_seed = 1, val_data = NULL) {
	if (is.null(formula) & is.null(dim) & is.null(x0)) {
		stop("Must pass one of 'formula', 'dim', 'x0'.")
	}
	if (!("list" %in% class(optimizer_args))) stop("'optimizer_args' must be a list.")
	if (NROW(optimizer) != 1 || !(optimizer %in% c('adaQN', 'SQN', 'oLBFGS'))) {
		stop("'optimizer' must be one of 'adaQN', 'SQN', 'oLBFGS'.")
	}
	if (!is.null(pos_class) & (NROW(pos_class) != 1 | !("character" %in% class(pos_class)))) {
		stop("'pos_class' must be a single string.")
	}
	if (!is.null(formula)) {
		if (!("formula" %in% class(formula))) {
			stop("'formula' must be a formula, e.g. 'y ~ x1 + x2'.")
		}
		if (is.null(pos_class)) stop("When using 'formula', must also pass 'pos_class'.")
		dim       <- NULL
		intercept <- NULL
	} else { intercept <- check.is.bool(intercept, "intercept") }
	if (is.null(x0)) {
		if (is.null(formula)) {
			dim <- check.positive.integer(dim, "dim") + intercept
			if (is.null(formula)) {
				set.seed(random_seed)
				x0 <- rnorm(dim, sd = 0.1)
			}
		}
	} else { random_seed <- NULL ; dim <- NROW(x0) }
	
	optimizer_args$x0       <- x0
	optimizer_args$grad_fun <- logistic_grad
	optimizer_args$pred_fun <- logistic_pred
	if (optimizer == "SQN") {
		optimizer_args$hess_vec_fun <- logistic_Hess_vec
	}
	if (optimizer == "adaQN") {
		optimizer_args$obj_fun      <- logistic_loss
		
		if (!is.null(val_data)) {
			
			if (!("list" %in% class(val_data)) || !("X" %in% names(val_data))) {
				stop("'val_data', if passed, must be a list with entries 'X', 'y', optionally 'w'.")
			}
			if (!is.null(formula)) {
				if (!("data.frame" %in% class(val_data$X))) {
					stop("'X' in validation set data must be a 'data.frame'.")
				}
				if (!is.null(val_data$y)) warning("'y' in validation data is ignored when passing formula.")
			} else {
				if (is.null(val_data$y)) stop("'y' in validation data cannot be missing when using formula.")
				if ("integer" %in% class(val_data$y)) val_data$y <- as.numeric(val_data$y)
				if (NROW(val_data$X) != NROW(val_data$y)) {
					stop("'y' in validation set data must have the same number of rows as 'X'.")
				}
				if (NCOL(val_data$y) != 1 || !("numeric" %in% class(val_data$y))) {
					stop("'y' in validation data must be a numeric vector.")
				}
			}
			
			if (!is.null(val_data$w) && (NROW(val_data$w) != NROW(val_data$X))) {
				stop("'w' in validation set data must have the same number of rows as 'X'.")
			}
			if ("integer" %in% class(val_data$w)) val_data$w <- as.numeric(val_data$w)
			if (!is.null(val_data$w) && (NCOL(val_data$w) != 1 || !("numeric" %in% class(val_data$w)))) {
				stop("'w' in validation data must be a numeric vector.")
			}
			
			optimizer_args$X_val <- val_data$X
			optimizer_args$y_val <- val_data$y
			optimizer_args$w_val <- val_data$w
		}
	}
	
	this <- list(coef = x0, initialized = FALSE, formula = formula, intercept = intercept,
				 pos_class = pos_class, random_seed = random_seed, colnames = NULL)
	if (!is.null(x0)) {
		this$stochQN <- switch(optimizer,
							   "oLBFGS" = do.call(oLBFGS, optimizer_args),
							   "SQN"    = do.call(SQN,    optimizer_args),
							   "adaQN"  = do.call(adaQN,  optimizer_args)
							   )
	} else {
		this$optimizer_args <- optimizer_args
		this$optimizer_name <- optimizer
	}
	
	class(this) <- "stoch_logistic"
	return(this)
}


#' @title Print general info about stochastic logistic regression object
#' @param x A `stoch_logistic` object as output by function `stochastic.logistic.regression`.
#' @param ... Not used.
#' @seealso \link{stochastic.logistic.regression}
#' @export
print.stoch_logistic <- function(x, ...) {
	if (!is.null(x$stochQN)) {
		if ("oLFBGS" %in% class(x$stochQN)) {
			opt_name <- "oLBFGS"
		} else if ("SQN" %in% class(x$stochQN)) {
			opt_name <- "SQN"
		} else if ("adaQN" %in% class(x$stochQN)) {
			opt_name <- "adaQN"
		} else {
			opt_name <- "INVALID_OBJECT"
		}
	} else {
		opt_name <- x$optimizer_name
	}
	
	cat(sprintf("Stochastic Logistic Regression - optimizer: %s\n\n", opt_name))
	if (!is.null(x$coef)) cat(sprintf("Number of features: %d\n", NROW(x$coef)))
	if (!is.null(x$formula)) {
		cat("Formula: ")
		print(x$formula)
	}
	if (!is.null(x$pos_class)) {
		cat(sprintf("Positive class: %s\n", x$pos_class))
	}
	cat(sprintf("Regularization strength: %f\n", x$lambda))
	if (!is.null(x$random_seed)) cat(sprintf("Random seed: %d\n", x$random_seed))
	if (!is.null(x$stochQN)) { niter <- get_iteration_number(x$stochQN) } else {niter <- 0}
	cat(sprintf("Number of partial fits: %d\n", niter))
}

#' @title Retrieve fitted coefficients from stochastic logistic regression object
#' @param object A `stoch_logistic` object as output by function `stochastic.logistic.regression`.
#' Must have already been fit to at least 1 batch of data.
#' @param ... Not used.
#' @return An (n x 1) matrix with the coefficients, in the same format as those from `glm`.
#' @seealso \link{stochastic.logistic.regression}
#' @export
coef.stoch_logistic <- function(object, ...) {
	if (!object$initialized) stop("Model has not been fit.")
	outp <- matrix(object$coef, nrow = NROW(object$coef))
	if (!is.null(object$colnames)) {
		row.names(outp) <- object$colnames
	}
	return(outp)
}

#' @title Print general info about stochastic logistic regression object
#' @description Same as `print` function. To check the fitted coefficients use function `coef`.
#' @param object A `stoch_logistic` object as output by function `stochastic.logistic.regression`.
#' @param ... Not used.
#' @seealso \link{coef.stoch_logistic} , \link{stochastic.logistic.regression}
#' @export
summary.stoch_logistic <- function(object, ...) {
	print.stoch_logistic(object)
}

#' @title Prediction function for stochastic logistic regression
#' @description Makes predictions for new data from the fitted model. Model have already
#' been fit to at least 1 batch of data.
#' @param object A `stoch_logistic` object as output by function `stochastic.logistic.regression`.
#' @param newdata New data on which to make predictions.
#' @param type Type of prediction to make. Can pass `prob` to get probabilities for the positive class,
#' or `class` to get the predicted class.
#' @param ... Not used.
#' @return A vector with the predicted classes or probabilities for `newdata`.
#' @seealso \link{stochastic.logistic.regression}
#' @export
predict.stoch_logistic <- function(object, newdata, type = "prob", ...) {
	if (!object$initialized) stop("Model has not been fit to any data.")
	if (!NROW(newdata)) stop("'newdata' has zero rows.")
	if (NROW(type) != 1 || !(type %in% c("class", "prob"))) {
		stop("'type' must be one of 'class' or 'prob'.")
	}
	if (is.null(object$formula)) {
		if (NCOL(newdata) != (NROW(object$coef) - object$intercept)) {
			stop("'newdata' has incorrect number of columns.")
		}
		if (object$intercept) {
			newdata <- cbind(rep(1, NROW(newdata)), newdata)
		}
		if ("data.frame" %in% class(newdata)) newdata <- as.matrix(newdata)
		if ("tibble" %in% class(newdata)) newdata <- as.matrix(newdata)
	} else {
		if (NROW(object$factor_cols)) {
			newdata[, object$factor_cols] <- as.data.frame(lapply(object$factor_cols,
														   function(cl, df, levs) factor(df[[cl]], levels = levs[[cl]]),
														   newdata[, object$factor_cols, drop = FALSE], object$factor_levs))
		}
		newdata <- model.matrix(object$formula, data = newdata)
	}
	pred <- 1 / (1 + exp(-as.numeric(newdata %*% object$coef)))
	if (type == "class") {
		pred <- pred >= .5
		if (!is.null(object$formula)) {
			pred <- ifelse(pred, object$pos_class, object$neg_class)
		} else {
			pred <- as.numeric(pred)
		}
	}
	return(pred)
}

#' @title Update stochastic logistic regression model with new batch of data
#' @description Perform a quasi-Newton iteration to update the model with new data.
#' @param logistic_model A `stoch_logistic` object as output by function `stochastic.logistic.regression`.
#' Will be modified in-place.
#' @param X Data with covariates. If passing a `data.frame`, the model object must have been initialized
#' with a formula, and `X` must also contain the target variable (`y`). If passing a matrix, must
#' also pass `y`. Note that whatever factor levels are present in the first batch of data, will be taken as the
#' whole factor levels.
#' @param y The target variable, when using matrices. Ignored when using formula.
#' @param w Sample weights (optional). If required, must pass them at every partial fit iteration.
#' @return No return value. Model object is updated in-place.
#' @seealso \link{stochastic.logistic.regression}
#' @export
partial_fit_logistic <- function(logistic_model, X, y = NULL, w = NULL) {
	
	### validate inputs
	if (!("stoch_logistic" %in% class(logistic_model))) {
		stop("Function is only applicable for 'stoch_logistic' objects.")
	}
	if (is.null(X)) stop("'X' cannot be missing.")
	if (!is.null(w) && (NROW(X) != NROW(w))) stop("'w' must have the same number of rows as 'X'.")
	if (!is.null(w) && (NCOL(w) > 1)) stop("'w' must be a 1-dimensional vector.")
	if ("matrix" %in% class(w)) w <- as.numeric(w)
	
	### initializa object if not already initialized
	if (!logistic_model$initialized) {
		this <- logistic_model
		if (any(sapply(X, function(x) "character" %in% class(x)))) {
			cols_str      <- names(X)[sapply(X, function(x) "character" %in% class(x))]
			X[, cols_str] <- as.data.frame(lapply(X[, cols_str, drop = FALSE], factor))
		}
		if (any(sapply(X, function(x) "factor" %in% class(x)))) {
			this$factor_cols <- names(X)[sapply(X, function(x) "factor" %in% class(x))]
		} else {
			this$factor_cols <- c()
			this$factor_levs <- list()
		}
		if (!is.null(this$formula)) {
			
			sample_X         <- model.matrix(this$formula, data = X[1, , drop = FALSE])
			this$target_col  <- attr(attr(terms(logistic_model$formula, data = sample_X), "factors"), "dimnames")[[1]][1]
			if (!(this$target_col %in% colnames(X))) stop("'X' does not contain target column.")
			if (!("factor" %in% class(X[[this$target_col]]))) stop("Target column in formula must be of class 'factor'.")
			if (NROW(levels(X[[this$target_col]])) != 2) stop("Target column must have 2 factor levels.")
			if (!(this$pos_class %in% levels(X[[this$target_col]]))) stop("Positive class is not a factor level in target column.")
			this$neg_class   <- setdiff(levels(X[[this$target_col]]), this$pos_class)
			this$factor_cols <- setdiff(this$factor_cols, this$target_col)
			this$factor_levs <- lapply(this$factor_cols, function(cl, df) levels(df[[cl]]), X)
			names(this$factor_levs) <- this$factor_cols
			
			if (!is.null(this$optimizer_args$X_val)) {
				if (NROW(this$factor_cols)) {
					this$optimizer_args$X_val[, this$factor_cols] <- as.data.frame(lapply(
																		this$factor_cols,
																		function(cl, df, levs) factor(df[[cl]], levels = levs[[cl]]),
																		this$optimizer_args$X_val[, this$factor_cols, drop = FALSE],
																		this$factor_levs))
				}
				if (!(this$target_col %in% colnames(this$optimizer_args$X_val))) {
					stop("'X' in validation data does not contain target column.")
				}
				this$optimizer_args$y_val <- as.numeric(this$optimizer_args$X_val[[this$target_col]] == this$pos_class)
				this$optimizer_args$X_val <- model.matrix(this$formula, data = this$optimizer_args$X_val)
			}
			
			dim           <- NCOL(sample_X)
			this$colnames <- colnames(sample_X)
			set.seed(this$random_seed)
			x0                     <- rnorm(dim, sd = 0.1)
			this$optimizer_args$x0 <- x0
			this$coef              <- x0
			this$stochQN           <- switch(this$optimizer_name,
											 "oLBFGS" = do.call(oLBFGS, this$optimizer_args),
											 "SQN"    = do.call(SQN,    this$optimizer_args),
											 "adaQN"  = do.call(adaQN,  this$optimizer_args)
											 )
			this$optimizer_args    <- NULL
			this$optimizer_name    <- NULL
		}
		
		this$initialized <- TRUE
		eval.parent(substitute(logistic_model <- this))
		logistic_model <- this
	}
	
	### Apply any transformations if required, validate conditional inputs
	if (!is.null(logistic_model$formula)) {
		if (!is.null(y)) stop("'y' is only ussed for non-formula models.")
		if (!("data.frame" %in% class(X))) stop("'X' must be a 'data.frame' when using formula.")
		if (NROW(logistic_model$factor_cols)) {
			X[, logistic_model$factor_cols] <- as.data.frame(lapply(logistic_model$factor_cols,
															 function(cl, df, levs) factor(df[[cl]], levels = levs[[cl]]),
															 X[, logistic_model$factor_cols, drop = FALSE],
															 logistic_model$factor_levs))
		}
		if (!(logistic_model$target_col %in% colnames(X))) stop("'X' does not contain target column.")
		y          <- as.numeric(X[[logistic_model$target_col]] == logistic_model$pos_class)
		X          <- model.matrix(logistic_model$formula, data = X)
	} else {
		if (NROW(X) != NROW(y)) stop("'X' and 'y' must have the same number of rows.")
		if (NCOL(y) > 1) stop("'y' must be a 1-dimensional vector.")
		if ("matrix" %in% class(y)) y <- as.numeric(y)
		
		if (logistic_model$intercept) {
			X <- cbind(rep(1.0, NROW(X)), X)
		}
		if (NCOL(X) != NROW(logistic_model$coef)) {
			stop("'X' has incorrect number of columns.")
		}
		if ("data.frame" %in% class(X)) X <- as.matrix(X)
		if ("tibble" %in% class(X))     X <- as.matrix(X)
	}
	
	### Pass processed inputs to optimizer
	partial_fit(logistic_model$stochQN, X, y, w)
}
