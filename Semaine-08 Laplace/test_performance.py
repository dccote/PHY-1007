import unittest
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import zoom
from enum import Enum
import math

from scalarfield import ScalarField

from utils import left, center, right, all
from solvers import LaplacianSolver, LaplacianSolverGPU

class PerformanceTestCase(unittest.TestCase):
	def setUp(self):
		print(self._testMethodName)

	def test_solve3D_CPU(self):
		pot = ScalarField(shape=(16,16,16))
		pot.solver = LaplacianSolver()

		pot.add_boundary_condition( ( -1, all, all), 10)
		pot.add_boundary_condition( (all,   0, all), 0)
		pot.add_boundary_condition( (all,  -1, all), 0)
		pot.add_boundary_condition( (all, all,   0), 0)
		pot.add_boundary_condition( (all, all,  -1), 0)
		pot.add_boundary_condition( ( 0 , all, all), 10)

		for i in [16,32,64,128,256,512]:
			start_time = time.time()
			pot.reset(shape=(i,i,i))
			pot.apply_conditions()
			pot.solve_laplace_by_relaxation(tolerance=1e-5)
			print(f"{pot.shape[0]}\t{time.time()-start_time:.3f}")
			pot.reset(np.array(pot.shape)*2)


	def test_solve3D_GPU(self):
		pot = ScalarField(shape=(16,16,16))
		pot.solver = LaplacianSolverGPU()

		pot.add_boundary_condition( ( -1, all, all), 10)
		pot.add_boundary_condition( (all,   0, all), 0)
		pot.add_boundary_condition( (all,  -1, all), 0)
		pot.add_boundary_condition( (all, all,   0), 0)
		pot.add_boundary_condition( (all, all,  -1), 0)
		pot.add_boundary_condition( ( 0 , all, all), 10)
		for i in [16,32,64,128,256,512]:

			start_time = time.time()
			pot.reset(shape=(i,i,i))
			pot.apply_conditions()
			pot.solve_laplace_by_relaxation(tolerance=1e-5)
			print(f"{pot.shape[0]}\t{time.time()-start_time:.3f}")

	def test_solve3D_CPU_refine(self):
		pot = ScalarField(shape=(16,16,16))
		pot.solver = LaplacianSolver()

		pot.add_boundary_condition( ( -1, all, all), 10)
		pot.add_boundary_condition( (all,   0, all), 0)
		pot.add_boundary_condition( (all,  -1, all), 0)
		pot.add_boundary_condition( (all, all,   0), 0)
		pot.add_boundary_condition( (all, all,  -1), 0)
		pot.add_boundary_condition( ( 0 , all, all), 10)
		for i in [16,32,64,128,256,512]:

			start_time = time.time()
			pot.reset(shape=(i,i,i))
			pot.apply_conditions()
			pot.solve_laplace_by_relaxation_with_refinements(tolerance=1e-5)
			print(f"{pot.shape[0]}\t{time.time()-start_time:.3f}")


if __name__ == "__main__":
	# unittest.main(defaultTest=['PerformanceTestCase.test_solve3D_CPU_refine'])
	unittest.main()

