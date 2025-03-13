import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import zoom
import math

try:
	import pyopencl as cl
	import pyopencl.array as cl_array
except:
	print("OpenCL not available. On Linux: sudo apt-get install python-pyopencl seems to work (Ubuntu ARM64 macOS).")
	cl = None
	cl_array = None

from utils import left, center, right, all

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

		h, w = field.shape
		global_size = field.shape
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
		d, h, w = field.shape
		global_size = field.shape
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
