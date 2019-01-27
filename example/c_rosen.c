/*	Example usage of SQN optimizer in package stochQN with Rosenbrock's (a.k.a. banana) function.

	Implementation of the functions taken from here:
	https://gist.github.com/Bismarrck/772d4ae4a8c5ddcd37b8
*/

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include "stochqn.h"

double rosen(double x[], int n)
{
	double dx = 0.0;
	double out = 0.0;
	for (int i = 0; i < n - 1; i ++) {
		dx = x[i + 1] - x[i] * x[i];
		out += 100.0 * dx * dx;
		dx = 1.0 - x[i];
		out += dx * dx;
	}
	return out;
}

void rosen_der(double x[], int n, double out[])
{
	out[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
	out[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
	
	double d1 = 0.0;
	double d2 = 0.0;
	double d3 = 0.0;
	
	for (int i = 1; i < n - 1; i ++) {
		d1 = 200.0 * (x[i] - x[i - 1] * x[i - 1]);
		d2 = 400.0 * (x[i + 1] - x[i] * x[i]) * x[i];
		d3 = 2.0 * (1.0 - x[i]);
		out[i] = d1 - d2 - d3;
	}
}

void rosen_hess_prod(double x[], double p[], int n, double out[])
{
	memset(out, 0, sizeof(double) * n);
	out[0] = (1200 * x[0] * x[0] - 400 * x[1] + 2.0) * p[0] - 400 * x[0] * p[0];
	out[n - 1] = -400.0 * x[n - 2] * p[n - 2] + 200.0 * p[n - 1];
	
	double d1 = 0.0;
	double d2 = 0.0;
	double d3 = 0.0;
	for (int i = 1; i < n - 1; i ++) {
		d1 = -400.0 * x[i - 1] * p[i - 1];
		d2 = (202 + 1200 * x[i] * x[i] - 400 * x[i + 1]) * p[i];
		d3 = 400.0 * x[i] * p[i + 1];
		out[i] = d1 + d2 - d3;
	}
}

void print_arr(double a[], int n)
{
	printf("x: [ ");
	for (int i = 0; i < n; i++){
		printf("%f ", a[i]);
	}
	printf(" ]\n");
} 

int main()
{
	/* Variables to optimize, allocations for gradient and hessian-vector product */
	int n = 4;
	double x[] = {1.3, 0.7, 0.8, 1.9, 1.2};
	double *grad = (double*) malloc(sizeof(double) * n);
	double *hess_vec = (double*) malloc(sizeof(double) * n);
	

	/* Optimizer parameters */
	size_t mem_size = 5;
	size_t bfgs_upd_freq = 3;
	double min_curvature = 0;
	int use_grad_diff = 0;
	int check_nan = 1;

	double step_size = 1e-3;
	double fun_val;
	double *req;
	double *req_vec;
	task_enum task;
	info_enum iter_info;
	int nthreads = 1;
	int variables_were_updated;

	printf("Optimization of Rosenbrock's function with SQN\n\n");
	printf("Initial function value: %6.4f\n", rosen(x, n));
	printf("Initial variable values:  ");
	print_arr(x, n);

	workspace_SQN* SQN = initialize_SQN(n, mem_size, bfgs_upd_freq, min_curvature, use_grad_diff, check_nan, nthreads);
	run_SQN(step_size, x, grad, hess_vec, &req, &req_vec, &task, SQN, &iter_info);

	while (SQN->niter < 200)
	{

		if (task == calc_grad)
		{
			rosen_der(req, n, grad);
		}

		else if (task == calc_hess_vec)
		{
			/* Note: when there is sample data, this calculation should be done on a larger batch than the gradient */
			rosen_hess_prod(req, req_vec, n, hess_vec);
		}

		variables_were_updated = run_SQN(step_size, x, grad, hess_vec, &req, &req_vec, &task, SQN, &iter_info);
		if (variables_were_updated && ( (SQN->niter + 1) % 10) == 0) { printf("Iteration %3d - f(x) = %6.4f\n", SQN->niter + 1, rosen(x, n)); }
	}

	printf("Optimization terminated - Final function value: %6.4f\n", rosen(x, n));
	printf("Final variable values:  ");
	print_arr(x, n);

	dealloc_SQN(SQN);
	free(grad);
	free(hess_vec);
	return 0;
}
