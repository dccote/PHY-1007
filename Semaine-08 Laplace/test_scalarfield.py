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

class PotentialTestCase(unittest.TestCase):
	def test_init(self):
		self.assertIsNotNone(ScalarField(shape=(32,32)))

	def test_show(self):
		ScalarField(shape=(32,32)).show(title=self._testMethodName)

	def test_conditions2D(self):
		pot = ScalarField(shape=(32,32))
		pot.add_boundary_condition( (all, 0), 10)

	def test_apply_conditions2D(self):
		pot = ScalarField(shape=(32,32))

		pot.add_boundary_condition((0, all), 10)
		pot.add_boundary_condition((-1, all), 5)
		pot.apply_conditions()
		pot.show(title=self._testMethodName)

	def test_solve2D(self):
		pot = ScalarField(shape=(32,32))

		pot.add_boundary_condition( (0, all), 10)
		pot.add_boundary_condition( (-1, all), 5)
		pot.apply_conditions()
		pot.solve_laplace_by_relaxation()

		pot.show(title=self._testMethodName)

	def test_solve2D_then_upscale(self):
		pot = ScalarField(shape=(8,8))
		pot.solver = LaplacianSolver()
		pot.add_boundary_condition( (all, 0), 0)
		pot.add_boundary_condition( (all, -1), 0)
		pot.add_boundary_condition( (0, all), 10)
		pot.add_boundary_condition( (-1, all), 5)

		pot.apply_conditions()
		start_time = time.time()		
		pot.solve_laplace_by_relaxation(tolerance=1e-6)
		pot.upscale(factor=8, order=2)
		pot.show(title=f"Upscaled solution")

		pot.solve_laplace_by_relaxation(tolerance=1e-6)
		pot.show(title=f"Actual solution")

	def test_solve2D_with_upscale(self):
		pot = ScalarField(shape=(8,8))
		pot.solver = LaplacianSolver()
		pot.add_boundary_condition( (all, 0), 0)
		pot.add_boundary_condition( (all, -1), 0)
		pot.add_boundary_condition( (0, all), 10)
		pot.add_boundary_condition( (-1, all), 5)

		start_time = time.time()		
		pot.solve_laplace_by_relaxation(tolerance=1e-6)
		pot.upscale(factor=16, order=0)
		it = pot.solve_laplace_by_relaxation(tolerance=1e-6)
		print(f"[{it}] With nearest refinement: {time.time()-start_time:.2f}")

		pot.reset(shape=(8,8))
		start_time = time.time()		
		pot.solve_laplace_by_relaxation(tolerance=1e-6)
		pot.upscale(factor=16, order=1)
		it = pot.solve_laplace_by_relaxation(tolerance=1e-6)
		print(f"[{it}] With linear refinement: {time.time()-start_time:.2f}")

		pot.reset(shape=(8,8))
		start_time = time.time()		
		pot.solve_laplace_by_relaxation(tolerance=1e-6)
		pot.upscale(factor=16, order=2)
		it = pot.solve_laplace_by_relaxation(tolerance=1e-6)
		print(f"[{it}] x16 With quadratic refinement: {time.time()-start_time:.2f}")
		pot.show(title=f"{pot.shape}")

		pot.reset(shape=(8,8))
		start_time = time.time()		
		pot.solve_laplace_by_relaxation(tolerance=1e-6)
		pot.upscale(factor=4, order=2)
		it1 = pot.solve_laplace_by_relaxation(tolerance=1e-6)
		pot.upscale(factor=4, order=2)
		it2 = pot.solve_laplace_by_relaxation(tolerance=1e-6)
		pot.upscale(factor=4, order=2)
		it3 = pot.solve_laplace_by_relaxation(tolerance=1e-6)
		print(f"[{it1+it2+it3}] x4x4x4 With quad refinement: {time.time()-start_time:.2f}")
		pot.show(title=f"{pot.shape}")

		# pot.reset()
		# pot.apply_conditions()
		# start_time = time.time()		
		# it = pot.solve_laplace_by_relaxation(tolerance=1e-6)
		# pot.show(title=f"{self._testMethodName} {pot.values.shape}")
		# print(f"[{it}] Without refinement: {time.time()-start_time:.2f}")

	def test_solve2D_CPU_vs_GPU(self):
		pot = ScalarField(shape=(32,32))
		pot.solver = LaplacianSolver()

		pot.add_boundary_condition( (0, all), 10)
		pot.add_boundary_condition( (-1, all), 5)
		pot.apply_conditions()
		pot.solver = LaplacianSolverGPU()

		pot.solve_laplace_by_relaxation()
		field_gpu = pot.values.copy()
		pot.show(title=f"{self._testMethodName} [GPU]")
		
		pot.values = np.zeros(shape=pot.shape, dtype=np.float32)
		pot.solver = LaplacianSolver()

		pot.solve_laplace_by_relaxation()
		field_cpu = pot.values.copy()
		self.assertTrue( (field_cpu-field_gpu).all() == 0)
		pot.show(title=f"{self._testMethodName} [CPU]")

	def test_solve2D_function_condition(self):
		pot = ScalarField(shape=(32,32))

		x = np.linspace(0,6.28,32)
		pot.add_boundary_condition( (0, all), 10*np.sin(x))
		pot.add_boundary_condition( (-1, all), -10*np.sin(x))
		pot.apply_conditions()
		pot.solve_laplace_by_relaxation()

		pot.show(title=self._testMethodName)

	def test_solve2D_funky_conditions(self):
		pot = ScalarField(shape=(32,32))

		pot.add_boundary_condition( (0, all), 10)
		pot.add_boundary_condition( (-1, all), 5)
		pot.add_boundary_condition( (10, slice(10,20)), 10)
		pot.add_boundary_condition( (slice(15,18), 20) , 10)
		pot.apply_conditions()
		pot.solve_laplace_by_relaxation()

		pot.show(title=self._testMethodName)

	def test_conditions3D(self):
		pot = ScalarField(shape=(32,32,32))
		pot.add_boundary_condition( (all, 0, 0), 10)
		pot.apply_conditions()

	def test_solve1D(self):
		pot = ScalarField(shape=(32))

		pot.add_boundary_condition( (0,), 10)
		pot.add_boundary_condition( (15,), 0)
		pot.add_boundary_condition( (-1,), 10)
		pot.apply_conditions()
		pot.solve_laplace_by_relaxation(tolerance=1e-8)

		pot.show(title=self._testMethodName)

	def test_solve3D(self):
		pot = ScalarField(shape=(64,64,64))

		pot.add_boundary_condition( ( -1, all, all), 10)
		pot.add_boundary_condition( (all,   0, all), 0)
		pot.add_boundary_condition( (all,  -1, all), 0)
		pot.add_boundary_condition( (all, all,   0), 0)
		pot.add_boundary_condition( (all, all,  -1), 0)
		pot.add_boundary_condition( ( 0 , all, all), 10)
		pot.apply_conditions()
		pot.solve_laplace_by_relaxation(tolerance=1e-7)

		pot.show(slices=(all, 31, all), title=self._testMethodName)


	def test_solve3D_GPU(self):
		pot = ScalarField(shape=(32,32,32))
		pot.solver = LaplacianSolverGPU()

		pot.add_boundary_condition( ( -1, all, all), 10)
		pot.add_boundary_condition( (all,   0, all), 0)
		pot.add_boundary_condition( (all,  -1, all), 0)
		pot.add_boundary_condition( (all, all,   0), 0)
		pot.add_boundary_condition( (all, all,  -1), 0)
		pot.add_boundary_condition( ( 0 , all, all), 10)
		pot.apply_conditions()
		pot.solve_laplace_by_relaxation(tolerance=1e-7)

		pot.show(slices=(all, 31, all), title=self._testMethodName)

if __name__ == "__main__":
	# unittest.main(defaultTest=['PotentialTestCase'])
	unittest.main()

