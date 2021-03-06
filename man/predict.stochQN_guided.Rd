% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/optimizers_guided.R
\name{predict.stochQN_guided}
\alias{predict.stochQN_guided}
\title{Predict function for stochastic optimizer object}
\usage{
\method{predict}{stochQN_guided}(object, newdata, ...)
}
\arguments{
\item{object}{Optimizer from this module as output by functions `oLBFGS`, `SQN`, `adaQN`. Must
have been constructed with a predict function.}

\item{newdata}{Data on which to make predictions (will be passed to the user-provided function).}

\item{...}{Additional arguments to pass to the user-provided predict function.}
}
\description{
Calls the user-defined predict function for an object
optimized through this package's functions.
}
\seealso{
\link{oLBFGS} , \link{SQN} , \link{adaQN}
}
