try:
	from setuptools import setup
	from setuptools import Extension
except:
	from distutils.core import setup
	from distutils.extension import Extension
import numpy as np, os
from findblas.distutils import build_ext_with_blas
from sys import platform

## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext_with_blas ):
	def build_extensions(self):
		from_rtd = os.environ.get('READTHEDOCS') == 'True'
		compiler = self.compiler.compiler_type
		if compiler == 'msvc': # visual studio
			for e in self.extensions:
				e.extra_compile_args += ['/O2', '/openmp']
		else: # everything else that cares about following standards
			for e in self.extensions:
				e.extra_compile_args += ['-O2', '-fopenmp', '-march=native', '-std=c99']
				e.extra_link_args += ['-fopenmp']
		for e in self.extensions:
			if from_rtd:
				e.define_macros += [("_FOR_RTD", None)]

			### remove this piece of code if you have macOS with GCC
			if platform[:3] == "dar":
				e.extra_compile_args = [arg for arg in e.extra_compile_args if arg != '-fopenmp']
				e.extra_link_args    = [arg for arg in e.extra_link_args    if arg != '-fopenmp']
		build_ext_with_blas.build_extensions(self)

setup(
	name  = "stochqn",
	packages = ["stochqn"],
	version = '0.2.8.2',
	description = 'Stochastic limited-memory quasi-Newton optimizers',
	author = 'David Cortes',
	author_email = 'david.cortes.rivera@gmail.com',
	url = 'https://github.com/david-cortes/stochQN',
	keywords = ['optimization', 'stochastic', 'quasi-Newton', 'SQN', 'adaQN', 'oLBFGS'],
	install_requires=[
		'numpy',
		'scipy',
		'scikit-learn',
		'cython',
		'findblas'
	],
	data_files = [('include', ['include/stochqn.h'])],
	cmdclass = {'build_ext': build_ext_subclass},
	ext_modules = [ Extension("stochqn._wrapper_double",
							sources = ["stochqn/wrapper_double.pyx", "src/stochqn.c"],
							include_dirs = [np.get_include(), "include"],
							define_macros = [("USE_DOUBLE", None), ("_FOR_PYTON", None)] ),
					Extension("stochqn._wrapper_float",
							sources = ["stochqn/wrapper_float.pyx", "src/stochqn.c"],
							include_dirs = [np.get_include(), "include"],
							define_macros = [("USE_FLOAT", None), ("_FOR_PYTON", None)] )]
	)
