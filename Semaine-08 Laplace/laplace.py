import unittest
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import zoom
from enum import Enum
import math

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
		self.values = np.zeros(shape=shape, dtype=np.float32)
		self.conditions = []
		self.solver = LaplacianSolver()

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

class LaplacianSolver:
	def solve_by_relaxation(self, field, tolerance):
		if field.values.ndim == 1:
			return self.solve1D_by_relaxation(field, tolerance)
		elif field.values.ndim == 2:
			return self.solve2D_by_relaxation(field, tolerance)
		elif field.values.ndim == 3:
			return self.solve3D_by_relaxation(field, tolerance)
		else:
			raise ValueError('Unable to manage dimension > 3')

	def solve1D_by_relaxation(self, field, tolerance):
		error = None
		field.apply_conditions()
		i = 0
		while error is None or error > tolerance:
			before_iteration = field.values.copy()
			field.values[center] = (field.values[left] + field.values[right])/2
			field.apply_conditions()
			error = np.std(field.values - before_iteration)
			error = np.std(field.values - before_iteration)
			i += 1

		return i

	def solve2D_by_relaxation(self, field, tolerance):
		error = None

		field.apply_conditions()
		i = 0
		while error is None or error > tolerance:
			if i % 100 == 0:
				before_iteration = field.values.copy()

			field.values[center, center] = (field.values[left,center] + field.values[right,center] +
									 field.values[center,left] + field.values[center,right])/4
			field.apply_conditions()
			if i % 100 == 0:
				error = np.std(field.values - before_iteration)
			i += 1

		return i

	def solve3D_by_relaxation(self, field, tolerance):
		error = None

		field.apply_conditions()
		i = 0
		while error is None or error > tolerance:
			if i % 100 == 0:
				before_iteration = field.values.copy()
			field.values[center, center, center] = (field.values[left, center, center] + field.values[center, left,center] +
											 field.values[center, center, left] + field.values[right,center,center] + 
											 field.values[center, right,center] + field.values[center, center, right])/6
			field.apply_conditions()
			if i % 100 == 0:
				error = np.std(field.values - before_iteration)
			i += 1

		return i

class LaplacianSolverGPU(LaplacianSolver):
	def __init__(self):
		super().__init__()
		self.platform = cl.get_platforms()[0]  # Select first platform
		self.device = self.platform.get_devices()[0]  # Select first device (GPU or CPU)
		self.context = cl.Context([self.device])  # Create OpenCL context
		self.queue = cl.CommandQueue(self.context)  # Create command queue
		self.program = cl.Program(self.context, self.kernel_code).build()

	def solve2D_by_relaxation(self, field, tolerance):
		field.apply_conditions()

		# Create OpenCL buffers: use cl.Array to benefit from operator overloading
		d_input = cl_array.to_device(self.queue, field.values)  # Copy data to GPU
		d_output = cl_array.empty_like(d_input)  # Create an empty GPU array

		h, w = field.values.shape
		global_size = field.values.shape
		size = math.prod(global_size)

		error = None
		i = 0
		while error is None or error > tolerance:
			# The calculation is sent to d_output, which I then use as the input for another iteration
			# This way, d_input becomes the output and I do not have to create an array each time.  This is very efficient.

			self.program.laplace2D(self.queue, global_size, None, d_input.data, d_output.data, np.int32(w))
			self.program.laplace2D(self.queue, global_size, None, d_output.data, d_input.data, np.int32(w))

			if i % 100 == 0:
				error = self.variance(d_output - d_input)
			i += 1

		field.values = d_input.get()
		return i

	def solve3D_by_relaxation(self, field, tolerance):
		field.apply_conditions()

		# Create OpenCL buffers
		d_input = cl_array.to_device(self.queue, field.values)  # Copy data to GPU
		d_output = cl_array.empty_like(d_input)  # Create an empty GPU array

		# Set up the execution parameters
		d, h, w = field.values.shape
		global_size = field.values.shape
		size = math.prod(global_size)

		error = None
		i = 0
		while error is None or error > tolerance:
			self.program.laplace3D(self.queue, global_size, None, d_input.data, d_output.data, np.int32(w), np.int32(h), np.int32(d))
			# The calculation is sent to d_output, which I then use as the input for another iteration
			# This way, d_input becomes the output and I do not have to create an array each time.  This is very efficient.
			self.program.laplace3D(self.queue, global_size, None, d_output.data, d_input.data, np.int32(w), np.int32(h), np.int32(d))

			if i % 100 == 0:
				error = self.variance(d_output - d_input)
			i += 1

		# Retrieve results
		field.values = d_output.get()
		return i

	def variance(self, d_diff):
		size = math.prod(d_diff.shape)

		mean_val = cl_array.sum(d_diff).get() / size  # Mean (transfers single float)
		d_diff_sq = (d_diff - mean_val) ** 2  # Element-wise (x - mean)^2
		variance_val = cl_array.sum(d_diff_sq).get() / size  # Variance (transfers single float)
		return np.sqrt(variance_val)  # Standard deviation (final sqrt)

	@property
	def kernel_code(self):
		kernel_code = """

		__kernel void laplace2D(__global float* input, __global float* output, int width) {
		    int x = get_global_id(0);
		    int y = get_global_id(1);
		    int index = y * width + x;

		    if (x == 0 || y == 0 || x == width-1 || y == width-1) {
		    	output[index] = input[index]; // Boundary is fixed
		    } else {
				output[index] = (input[index-1] + input[index+1] + input[index-width] + input[index+width])/4;
			}
		}

		__kernel void laplace3D(__global float* input, __global float* output, int width, int height, int depth) {
		    int x = get_global_id(0);
		    int y = get_global_id(1);
		    int z = get_global_id(2);

		    int index = z * (width * height) + y * width + x;

		    if (x == 0 || y == 0 || z == 0 || x == width-1 || y == height-1 || z == depth-1) {
		    	output[index] = input[index];
		    } else {
				output[index] = (input[index-1] + input[index+1] + input[index-width] + input[index+width] + input[index-width*height] + input[index+width*height])/6;
			}
		}


		__kernel void zoom2D_nearest_neighbour(__global float* input, __global float* output, int width, int height) {
		    int x = get_global_id(0);
		    int y = get_global_id(1);

		    int index_src = y * width + x;

		    int index_dest1 = (2*y) * (2*width) + 2*x;
		    int index_dest2 = (2*y) * (2*width) + 2*x + 1;
		    int index_dest3 = (2*y) * (2*width) + 2*x + 2*width;
		    int index_dest4 = (2*y) * (2*width) + 2*x + 2*width + 1;

	    	output[index_dest1] = input[index_src];
	    	output[index_dest2] = input[index_src];
	    	output[index_dest3] = input[index_src];
	    	output[index_dest4] = input[index_src];
		}

		__kernel void copy(__global float* input, __global float* output, int width) {
		    int x = get_global_id(0);
		    int y = get_global_id(1);

		    int index = y * width + x;

	    	output[index] = input[index];
		}

		"""
		return kernel_code


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
		print(f"[{it}] With NN refinement: {time.time()-start_time:.2f}")

		pot.reset(shape=(8,8))
		start_time = time.time()		
		pot.solve_laplace_by_relaxation(tolerance=1e-6)
		pot.upscale(factor=16, order=1)
		it = pot.solve_laplace_by_relaxation(tolerance=1e-6)
		print(f"[{it}] With lin refinement: {time.time()-start_time:.2f}")

		pot.reset(shape=(8,8))
		start_time = time.time()		
		pot.solve_laplace_by_relaxation(tolerance=1e-6)
		pot.upscale(factor=16, order=2)
		it = pot.solve_laplace_by_relaxation(tolerance=1e-6)
		print(f"[{it}] x16 With quad refinement: {time.time()-start_time:.2f}")
		pot.show(title=f"{pot.values.shape}", block=True)

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
		pot.show(title=f"{pot.values.shape}", block=True)

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
		
		pot.values = np.zeros(shape=pot.values.shape, dtype=np.float32)
		pot.solver = LaplacianSolver()

		pot.solve_laplace_by_relaxation()
		field_cpu = pot.values.copy()
		self.assertTrue( (field_cpu-field_gpu).all() == 0)
		pot.show(title=f"{self._testMethodName} [CPU]")

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


if __name__ == "__main__":
	unittest.main(defaultTest=['PotentialTestCase'])




