"""
Module for solving the Laplace equation using both CPU-based and GPU-accelerated solvers.

This module provides:
1. `LaplacianSolver`: Implements relaxation methods for solving the Laplace equation 
   in 1D, 2D, and 3D using iterative numerical approaches.
2. `LaplacianSolverGPU`: Extends `LaplacianSolver` to utilize GPU acceleration via 
   OpenCL, significantly improving performance for large arrays (no benefits for small 2D)

The GPU solver requires `pyopencl` and executes OpenCL kernels for parallelized 
computations on compatible hardware.

Dependencies:
- NumPy
- SciPy
- PyOpenCL (optional, required for `LaplacianSolverGPU`)
"""

import math
import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
except ImportError:
    print(
        "OpenCL not available. On Linux: sudo apt-get install python-pyopencl "
        "seems to work (Ubuntu ARM64, macOS)."
    )
    cl = None
    cl_array = None

from utils import left, center, right


class LaplacianSolver:
    """
    A CPU-based solver for the Laplace equation using relaxation methods.

    This class provides methods for solving the Laplace equation in
    1D, 2D, and 3D using iterative relaxation. The solver updates
    each grid point based on its neighboring values until convergence
    is reached.
    """

    def solve_by_relaxation(self, field, tolerance):
        """
        Selects the appropriate relaxation solver based on the field's dimensionality.

        Parameters:
        field (ScalarField): The field to be solved by the method of relaxation.
                             May contain an initial guess.
        tolerance (float): Convergence threshold.

        Returns:
        int: The number of iterations required for convergence.
        """

        if field.values.ndim not in [1, 2, 3]:
            raise ValueError("Unable to manage dimension > 3")

        if field.values.ndim == 1:
            return self.solve1D_by_relaxation(field, tolerance)

        if field.values.ndim == 2:
            return self.solve2D_by_relaxation(field, tolerance)

        return self.solve3D_by_relaxation(field, tolerance)

    def solve1D_by_relaxation(self, field, tolerance):  # pylint: disable=invalid-name
        """
        Solves the Laplace equation in 1D using iterative relaxation.

        Parameters:
        field (ScalarField): The 1D field to be solved. May contain an initial guess.
        tolerance (float): Convergence threshold.

        Returns:
        int: The number of iterations performed.
        """
        error = None
        field.apply_conditions()
        i = 0
        while error is None or error > tolerance:
            before_iteration = field.values.copy()
            field.values[center] = (field.values[left] + field.values[right]) / 2
            field.apply_conditions()
            error = np.std(field.values - before_iteration)
            i += 1

        return i

    def solve2D_by_relaxation(self, field, tolerance):  # pylint: disable=invalid-name
        """
        Solves the Laplace equation in 2D using iterative relaxation.

        Parameters:
        field (ScalarField): The 2D field to be solved. May contain an initial guess.
        tolerance (float): Convergence threshold.

        Returns:
        int: The number of iterations performed.
        """
        error = None
        field.apply_conditions()
        i = 0
        while error is None or error > tolerance:
            if i % 100 == 0:
                before_iteration = field.values.copy()

            field.values[center, center] = (
                field.values[left, center]
                + field.values[right, center]
                + field.values[center, left]
                + field.values[center, right]
            ) / 4
            field.apply_conditions()
            if i % 100 == 0:
                error = np.std(field.values - before_iteration)
            i += 1

        return i

    def solve3D_by_relaxation(self, field, tolerance):  # pylint: disable=invalid-name
        """
        Solves the Laplace equation in 3D using iterative relaxation.

        Parameters:
        field (ScalarField): The 3D field to be solved. May contain an initial guess.
        tolerance (float): Convergence threshold.

        Returns:
        int: The number of iterations performed.
        """
        error = None
        field.apply_conditions()
        i = 0
        while error is None or error > tolerance:
            if i % 100 == 0:
                before_iteration = field.values.copy()
            field.values[center, center, center] = (
                field.values[left, center, center]
                + field.values[center, left, center]
                + field.values[center, center, left]
                + field.values[right, center, center]
                + field.values[center, right, center]
                + field.values[center, center, right]
            ) / 6
            field.apply_conditions()
            if i % 100 == 0:
                error = np.std(field.values - before_iteration)
            i += 1

        return i


class LaplacianSolverGPU(LaplacianSolver):
    """
    A GPU-accelerated solver for the Laplace equation using OpenCL.

    This class extends `LaplacianSolver` and offloads computation to
    a GPU using OpenCL, significantly improving performance.
    """

    def __init__(self):
        """
        Initializes the OpenCL context, device, queue, and kernel program.
        """
        super().__init__()
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        self.program = cl.Program(self.context, self.kernel_code).build()

    def solve2D_by_relaxation(self, field, tolerance):  # pylint: disable=invalid-name
        """
        Solves the Laplace equation in 2D using OpenCL on a GPU.

        Parameters:
        field (ScalarField): The 2D field to be solved. May contain an initial guess.
        tolerance (float): Convergence threshold.

        Returns:
        int: The number of iterations performed.
        """
        field.apply_conditions()
        d_input = cl_array.to_device(self.queue, field.values)
        d_output = cl_array.empty_like(d_input)

        global_size = field.shape
        error = None
        i = 0
        while error is None or error > tolerance:
            self.program.laplace2D(
                self.queue,
                global_size,
                None,
                d_input.data,
                d_output.data,
                np.int32(field.shape[1]),
            )
            self.program.laplace2D(
                self.queue,
                global_size,
                None,
                d_output.data,
                d_input.data,
                np.int32(field.shape[1]),
            )

            if i % 100 == 0:
                error = self.variance(d_output - d_input)
            i += 1

        field.values = d_input.get()
        return i

    def variance(self, d_diff):
        """
        Computes the standard deviation of the difference between two fields.

        Parameters:
        d_diff (cl_array.Array): The difference array.

        Returns:
        float: The standard deviation.
        """
        size = math.prod(d_diff.shape)
        mean_val = cl_array.sum(d_diff).get() / size
        d_diff_sq = (d_diff - mean_val) ** 2
        variance_val = cl_array.sum(d_diff_sq).get() / size
        return np.sqrt(variance_val)

    @property
    def kernel_code(self):
        """
        Returns the OpenCL kernel code for solving the Laplace equation.

        Returns:
        str: The OpenCL kernel code.
        """
        return """

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

