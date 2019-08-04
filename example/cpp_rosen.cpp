/*	Example usage of SQN optimizer in package stochQN with Rosenbrock's (a.k.a. banana) function.
	Implementation of the functions taken from here:
	https://gist.github.com/Bismarrck/772d4ae4a8c5ddcd37b8
*/

#include <vector>
#include <iostream>
#include <iomanip>
#include <stddef.h>
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
	std::fill(out, out + n, 0);
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
	std::cout << "x: [ ";
	for (int i = 0; i < n; i++)
		std::cout << std::fixed << std::setprecision(3) << a[i] << " ";
	std::cout << " ]" << std::endl;
} 

int main()
{
	/* Variables to optimize, allocations for gradient and hessian-vector product */
	int n = 4;
	std::vector<double>x{1.3, 0.7, 0.8, 1.9, 1.2};
	std::vector<double> grad(n);
	std::vector<double> hess_vec(n);
	

	/* Optimizer parameters */
	size_t mem_size = 5;
	size_t bfgs_upd_freq = 3;
	double min_curvature = 0;

	double step_size = 1e-3;
	double fun_val;
	int variables_were_updated;

	std::cout << "Optimization of Rosenbrock's function with SQN\n" << std::endl;
	std::cout << "Initial function value: " << std::setw(7) << std::setprecision(4) << rosen(x.data(), n) << std::endl;
	std::cout << "Initial variable values:  ";
	print_arr(x.data(), n);

	SQN SQN_obj(n, mem_size, bfgs_upd_freq, min_curvature);
	SQN_obj.run(step_size, x.data(), grad.data(), hess_vec.data());

	while (SQN_obj.get_n_iter() < 200)
	{

		if (SQN_obj.get_task() == calc_grad)
		{
			rosen_der(SQN_obj.get_req(), n, grad.data());
		}

		else if (SQN_obj.get_task() == calc_hess_vec)
		{
			/* Note: when there is sample data, this calculation should be done on a larger batch than the gradient */
			rosen_hess_prod(SQN_obj.get_req(), SQN_obj.get_req_vec(), n, hess_vec.data());
		}

		variables_were_updated = SQN_obj.run(step_size, x.data(), grad.data(), hess_vec.data());
		if (variables_were_updated && ( (SQN_obj.get_n_iter() + 1) % 10) == 0)
		{
			std::cout << "Iteration " << std::setw(3) << (SQN_obj.get_n_iter() + 1)
			<< " - f(x) = " << std::setprecision(4) << std::setw(6) << rosen(x.data(), n) << std::endl;
		}
	}

	std::cout << "Optimization terminated - Final function value: ";
	std::cout << std::setw(7) << std::setprecision(4) << rosen(x.data(), n) << std::endl;
	std::cout << "Final variable values:  ";
	print_arr(x.data(), n);

	return EXIT_SUCCESS;
}
