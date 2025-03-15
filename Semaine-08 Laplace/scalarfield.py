import unittest
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import zoom
from enum import Enum
import math
from solvers import LaplacianSolver, LaplacianSolverGPU
from utils import left, center, right, all

class ScalarField:
	def __init__(self, shape):
		self.values = np.zeros(shape=shape, dtype=np.float32)
		self.conditions = []
		self.solver = LaplacianSolver()

	@property
	def shape(self):
		return self.values.shape
		
	def reset(self, shape=None):
		if shape is None:
			shape = self.values.shape

		self.values = np.zeros(shape=shape, dtype=np.float32)

	def upscale(self, factor=8, order=2):
		self.values = zoom(self.values, factor, order=order)

	def add_boundary_condition(self, index_or_slices, value_or_values):
		self.conditions.append((index_or_slices, value_or_values))

	def apply_conditions(self):
		for index_or_slices, value in self.conditions:
			self.values[*index_or_slices] = value

	def solve_laplace_by_relaxation_with_refinements(self, tolerance=1e-7):
		# print(f"Demanded: {self.shape}")
		final_shape = self.shape
		iterations = 1
		while self.shape[0] > 16:
			self.reset(np.array(self.shape)//8)
			iterations += 1

		# print(f"Required {iterations}, starting with: {self.shape}")
		for i in range(iterations):
			self.solver.solve_by_relaxation(self, tolerance=tolerance)
			if i != iterations-1:
				self.upscale(factor=8, order=2)
			

		return self

	def solve_laplace_by_relaxation(self, tolerance=1e-7):
		return self.solver.solve_by_relaxation(self, tolerance=tolerance)

	def show(self, slices=None, title=None, block=False):
		plt.clf()
		plt.title(title)
		if self.values.ndim == 1:
			plt.plot(self.values)
		else:
			if slices is not None:
				plt.imshow(self.values[*slices])
			else:
				plt.imshow(self.values)

		if block:
			plt.show()
		else:
			plt.pause(0.5)





