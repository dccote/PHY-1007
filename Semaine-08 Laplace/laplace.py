import unittest
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import zoom

try:
	import pyopencl as cl
	import pyopencl.array as cl_array
except:
	print("OpenCL not available. On Linux: sudo apt-get install python-pyopencl seems to work (Ubuntu ARM64 macOS).")
	cl = None
	cl_array = None


class ScalarField2D:
	def __init__(self, shape):
		self.field = np.zeros(shape=shape, dtype=np.float32)
		self.conditions = []

	def add_boundary_condition(self, index0_or_slice0, index1_or_slice1, value_or_values):
		self.conditions.append((index0_or_slice0, index1_or_slice1, value_or_values))

	def apply_conditions(self):
		for slice0, slice1, value in self.conditions:
			self.field[slice0, slice1] = value

	def solve_laplace_by_relaxation(self, tolerance=1e-7):
		error = None
		while error is None or error > tolerance:
			before_iteration = self.field.copy()
			self.field[1:-1,1:-1] = (self.field[:-2,1:-1]+ self.field[2:,1:-1]+ self.field[1:-1,:-2]+ self.field[1:-1,2:])/4
			self.apply_conditions()
			error = np.std(self.field - before_iteration)

		return self.field

	def show(self, title=None, block=False):
		plt.imshow(self.field)
		if block:
			plt.show()
		else:
			plt.pause(0.5)

class ScalarField3D:
	def __init__(self, shape):
		self.field = np.zeros(shape=shape, dtype=np.float32)
		self.conditions = []

	def add_boundary_condition(self, index0_or_slice0, index1_or_slice1, index2_or_slice2, value_or_values):
		self.conditions.append((index0_or_slice0, index1_or_slice1, index2_or_slice2, value_or_values))

	def apply_conditions(self):
		for slice0, slice1, slice2, value in self.conditions:
			self.field[slice0, slice1, slice2] = value

	def solve_laplace_by_relaxation(self, tolerance=1e-7):
		error = None
		
		left = slice(0,-2)     # [0:-2]
		center = slice(1,-1)   # [1:-1]
		right = slice(2, None) # [2:  ]

		iteration = 0
		while error is None or error > tolerance:
			if iteration % 20 == 0:
				before_iteration = self.field.copy()
			self.field[center, center, center] = (self.field[left, center, center] + self.field[center, left,center] +
												  self.field[center, center, left] + self.field[right,center,center] + 
												  self.field[center, right,center] + self.field[center, center, right])/6
			self.apply_conditions()
			if iteration % 20 == 0:
				error = np.std(self.field - before_iteration)
			iteration += 1

		return self.field

	def show(self, title=None, block=False):
		n0, n1, n2 = self.field.shape

		for i in range(n0):
			plt.imshow(self.field[i,:,:])
			if block:
				plt.show()
			else:
				plt.pause(0.5)

class Potential2DTestCase(unittest.TestCase):
	def test_init(self):
		self.assertIsNotNone(ScalarField2D(shape=(32,32)))

	def test_conditions(self):
		pot = ScalarField2D(shape=(32,32))
		pot.add_boundary_condition(slice(None, None), slice(0), 10)

	def test_apply_conditions(self):
		pot = ScalarField2D(shape=(32,32))

		pot.add_boundary_condition(0, slice(None, None), 10)
		pot.add_boundary_condition(-1, slice(None, None), 5)
		pot.apply_conditions()
		pot.show()

	def test_solve(self):
		pot = ScalarField2D(shape=(32,32))

		pot.add_boundary_condition(0, slice(None, None), 10)
		pot.add_boundary_condition(-1, slice(None, None), 5)
		pot.apply_conditions()
		pot.solve_laplace_by_relaxation()

		pot.show()

	def test_solve_function_condition(self):
		pot = ScalarField2D(shape=(32,32))

		x = np.linspace(0,6.28,32)
		pot.add_boundary_condition(0, slice(0,32), 10*np.sin(x))
		pot.add_boundary_condition(slice(0,32), 0, 10*np.sin(x))
		pot.apply_conditions()
		pot.solve_laplace_by_relaxation()

		pot.show()

	def test_solve_funky_conditions(self):
		pot = ScalarField2D(shape=(32,32))

		pot.add_boundary_condition(0, slice(None, None), 10)
		pot.add_boundary_condition(-1, slice(None, None), 5)
		pot.add_boundary_condition(10, slice(10,20), 10)
		pot.add_boundary_condition(slice(15,18), 20 , 10)
		pot.apply_conditions()
		pot.solve_laplace_by_relaxation()

		pot.show()

	def test_show(self):
		ScalarField2D(shape=(32,32)).show()


class Potential3DTestCase(unittest.TestCase):
	def test_init(self):
		self.assertIsNotNone(ScalarField3D(shape=(32,32,32)))

	def test_conditions(self):
		pot = ScalarField3D(shape=(32,32,32))
		pot.add_boundary_condition(slice(None, None), slice(0), 0, 10)

	def test_solve(self):
		pot = ScalarField3D(shape=(32,32,32))

		pot.add_boundary_condition(0, slice(None, None), slice(None, None), 10)
		pot.add_boundary_condition(slice(None, None), -1, slice(None, None), 5)
		pot.add_boundary_condition(16, slice(None, None), slice(None, None), 5)
		pot.apply_conditions()
		pot.solve_laplace_by_relaxation()

		pot.show(block=False)


if __name__ == "__main__":
	# unittest.main(defaultTest=["OpenCLArrayTestCase.test05_laplace2d_grid_refinement","OpenCLArrayTestCase.test01_2Dopencl","ArrayManipulationTestCase.test12_laplace_with_finer_and_finer_grid","ArrayManipulationTestCase.test10_laplace_initial_condition_fct"])
	unittest.main(defaultTest=['Potential3DTestCase'])




