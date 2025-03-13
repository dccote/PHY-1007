import unittest
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import zoom
from enum import Enum

try:
	import pyopencl as cl
	import pyopencl.array as cl_array
except:
	print("OpenCL not available. On Linux: sudo apt-get install python-pyopencl seems to work (Ubuntu ARM64 macOS).")
	cl = None
	cl_array = None

left   = slice(0,-2)
center = slice(1,-1)
right  = slice(2, None)
all    = slice(None, None)

class ScalarField:
	def __init__(self, shape):
		self.field = np.zeros(shape=shape, dtype=np.float32)
		self.conditions = []

	def add_boundary_condition(self, index_or_slices, value_or_values):
		self.conditions.append((index_or_slices, value_or_values))

	def apply_conditions(self):
		for index_or_slices, value in self.conditions:
			self.field[*index_or_slices] = value

	def solve_laplace_by_relaxation(self, tolerance=1e-7):
		error = None

		self.apply_conditions()

		while error is None or error > tolerance:
			before_iteration = self.field.copy()
			if self.field.ndim == 1:
				self.field[center] = (self.field[left] + self.field[right])/2
			elif self.field.ndim == 2:
				self.field[center, center] = (self.field[left,center] + self.field[right,center] +
											  self.field[center,left] + self.field[center,right])/4
			elif self.field.ndim == 3:
				self.field[center, center, center] = (self.field[left, center, center] + self.field[center, left,center] +
												  self.field[center, center, left] + self.field[right,center,center] + 
												  self.field[center, right,center] + self.field[center, center, right])/6
			else:
				raise ValueError('Unable to manage dimension > 3')

			self.apply_conditions()
			error = np.std(self.field - before_iteration)

		return self.field

	def show(self, slices=None, title=None, block=False):
		plt.clf()
		plt.title(title)
		if self.field.ndim == 1:
			plt.plot(self.field)
		else:
			if slices is not None:
				plt.imshow(self.field[*slices])
			else:
				plt.imshow(self.field)

		if block:
			plt.show()
		else:
			plt.pause(0.5)

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

	def test_solve2D_function_condition(self):
		pot = ScalarField(shape=(32,32))

		x = np.linspace(0,6.28,32)
		pot.add_boundary_condition( (0, all), 10*np.sin(x))
		pot.add_boundary_condition( (all, 0), 10*np.sin(x))
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

	def test_solve1D(self):
		pot = ScalarField(shape=(32))

		pot.add_boundary_condition( (0,), 10)
		pot.add_boundary_condition( (15,), 0)
		pot.add_boundary_condition( (-1,), 10)
		pot.apply_conditions()
		pot.solve_laplace_by_relaxation(tolerance=1e-8)

		pot.show(title=self._testMethodName)

	def test_solve3D(self):
		pot = ScalarField(shape=(32,32,32))

		pot.add_boundary_condition( ( -1, all, all), 0)
		pot.add_boundary_condition( (all,   0, all), 0)
		pot.add_boundary_condition( (all,  -1, all), 0)
		pot.add_boundary_condition( (all, all,   0), 0)
		pot.add_boundary_condition( (all, all,  -1), 0)
		pot.add_boundary_condition( ( 0 , all, all), 10)
		pot.apply_conditions()
		pot.solve_laplace_by_relaxation(tolerance=1e-8)

		pot.show(slices=(all, 15, all), title=self._testMethodName, block=True)


if __name__ == "__main__":
	unittest.main(defaultTest=['PotentialTestCase'])




