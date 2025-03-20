"""
Tests for performance CPU vs GPU

The different solvers for different conditions are tested.

"""
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
    """
    A unittest TestCase for evaluating the performance of solving the 3D Laplace equation
    using CPU and GPU solvers with different grid sizes.
    """
    def setUp(self):
        """
        Set up the test environment by printing the name of the test being executed.
        """
        print(self._testMethodName)

    def test_solve3D_CPU(self):
        """
        Tests the performance of solving the 3D Laplace equation using a CPU-based solver.
        Runs simulations for increasing grid sizes and records the execution time.
        """
        pot = ScalarField(shape=(16, 16, 16))
        pot.solver = LaplacianSolver()

        # Define boundary conditions
        pot.add_boundary_condition((-1, all, all), 10)
        pot.add_boundary_condition((all, 0, all), 0)
        pot.add_boundary_condition((all, -1, all), 0)
        pot.add_boundary_condition((all, all, 0), 0)
        pot.add_boundary_condition((all, all, -1), 0)
        pot.add_boundary_condition((0, all, all), 10)

        # Test different grid sizes
        for i in [16, 32, 64, 128]:
            start_time = time.time()
            pot.reset(shape=(i, i, i))
            pot.apply_conditions()
            pot.solve_laplace_by_relaxation(tolerance=1e-5)
            pot.show(slices=(all, i // 2, all))
            print(f"{pot.shape[0]}\t{time.time()-start_time:.3f}")
            pot.reset(np.array(pot.shape) * 2)

    def test_solve3D_GPU(self):
        """
        Tests the performance of solving the 3D Laplace equation using a GPU-based solver.
        Runs simulations for increasing grid sizes and records the execution time.
        """
        pot = ScalarField(shape=(16, 16, 16))
        pot.solver = LaplacianSolverGPU()

        # Define boundary conditions
        pot.add_boundary_condition((-1, all, all), 10)
        pot.add_boundary_condition((all, 0, all), 0)
        pot.add_boundary_condition((all, -1, all), 0)
        pot.add_boundary_condition((all, all, 0), 0)
        pot.add_boundary_condition((all, all, -1), 0)
        pot.add_boundary_condition((0, all, all), 10)
        
        # Test different grid sizes
        for i in [16, 32, 64, 128, 256, 512]:
            start_time = time.time()
            pot.reset(shape=(i, i, i))
            pot.apply_conditions()
            pot.solve_laplace_by_relaxation(tolerance=1e-5)
            pot.show(slices=(all, i // 2, all))
            print(f"{pot.shape[0]}\t{time.time()-start_time:.3f}")

    def test_solve3D_CPU_refine(self):
        """
        Tests the performance of solving the 3D Laplace equation using a CPU-based solver
        with grid refinement techniques. Runs simulations for increasing grid sizes and records
        the execution time.
        """
        pot = ScalarField(shape=(16, 16, 16))
        pot.solver = LaplacianSolver()

        # Define boundary conditions
        pot.add_boundary_condition((-1, all, all), 10)
        pot.add_boundary_condition((all, 0, all), 0)
        pot.add_boundary_condition((all, -1, all), 0)
        pot.add_boundary_condition((all, all, 0), 0)
        pot.add_boundary_condition((all, all, -1), 0)
        pot.add_boundary_condition((0, all, all), 10)
        
        # Test different grid sizes
        for i in [16, 32, 64, 128, 256, 512]:
            start_time = time.time()
            pot.reset(shape=(i, i, i))
            pot.apply_conditions()
            pot.solve_laplace_by_relaxation_with_refinements(tolerance=1e-5)
            pot.show(slices=(all, i // 2, all))
            print(f"{pot.shape[0]}\t{time.time()-start_time:.3f}")

    def test_solve3D_GPU_with_refinements(self):
        """
        Tests the performance of solving the 3D Laplace equation using a GPU-based solver
        with grid refinement techniques. Runs simulations for increasing grid sizes and records
        the execution time.
        """
        pot = ScalarField(shape=(16, 16, 16))
        pot.solver = LaplacianSolverGPU()

        # Define boundary conditions
        pot.add_boundary_condition((-1, all, all), 10)
        pot.add_boundary_condition((all, 0, all), 0)
        pot.add_boundary_condition((all, -1, all), 0)
        pot.add_boundary_condition((all, all, 0), 0)
        pot.add_boundary_condition((all, all, -1), 0)
        pot.add_boundary_condition((0, all, all), 10)
        
        # Test different grid sizes
        for i in [16, 32, 64, 128, 256, 512]:
            start_time = time.time()
            pot.reset(shape=(i, i, i))
            pot.apply_conditions()
            pot.solve_laplace_by_relaxation_with_refinements(
                factors=[8], tolerance=1e-5
            )
            print(f"{pot.shape[0]}\t{time.time()-start_time:.3f}")

if __name__ == "__main__":
    unittest.main(defaultTest=["PerformanceTestCase.test_solve3D_GPU"])
