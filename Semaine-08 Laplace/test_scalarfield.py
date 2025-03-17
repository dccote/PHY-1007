"""
Module for testing the ScalarField class and its methods using unittest.

This module includes various test cases to verify the functionality of 
ScalarField, particularly in solving the Laplace equation under different 
boundary conditions, resolutions, and computational methods (CPU vs GPU).
"""

import unittest
import numpy as np
from scalarfield import ScalarField
from solvers import LaplacianSolver, LaplacianSolverGPU
from utils import all

class PotentialTestCase(unittest.TestCase):
    """
    Unit tests for the ScalarField class.

    These tests evaluate different aspects of the ScalarField, including
    initialization, boundary condition application, Laplace equation solving,
    and visualization. Tests are performed for 1d, 2d, and 3d fields,
    with comparisons between CPU and GPU solvers.
    """

    def test_init(self):
        """Test that a ScalarField instance is initialized properly."""
        self.assertIsNotNone(ScalarField(shape=(32, 32)))

    def test_show(self):
        """Test the visualization of an empty ScalarField."""
        ScalarField(shape=(32, 32)).show(title=self._testMethodName)

    def test_conditions_2d(self):
        """Test adding a simple boundary condition in 2d."""
        pot = ScalarField(shape=(32, 32))
        pot.add_boundary_condition((all, 0), 10)

    def test_apply_conditions_2d(self):
        """Test applying boundary conditions in 2d."""
        pot = ScalarField(shape=(32, 32))
        pot.add_boundary_condition((0, all), 10)
        pot.add_boundary_condition((-1, all), 5)
        pot.apply_conditions()
        pot.show(title=self._testMethodName)

    def test_solve_2d(self):
        """Test solving the Laplace equation in 2d using relaxation."""
        pot = ScalarField(shape=(32, 32))
        pot.add_boundary_condition((0, all), 10)
        pot.add_boundary_condition((-1, all), 5)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation()
        pot.show(title=self._testMethodName)

    def test_solve_2d_then_upscale(self):
        """Test solving in 2d, then upscaling the solution and refining it."""
        pot = ScalarField(shape=(8, 8))
        pot.solver = LaplacianSolver()
        pot.add_boundary_condition((all, 0), 0)
        pot.add_boundary_condition((all, -1), 0)
        pot.add_boundary_condition((0, all), 10)
        pot.add_boundary_condition((-1, all), 5)

        pot.apply_conditions()
        pot.solve_laplace_by_relaxation(tolerance=1e-6)
        pot.upscale(factor=8, order=2)
        pot.show(title="Upscaled solution")

        pot.solve_laplace_by_relaxation(tolerance=1e-6)
        pot.show(title="Actual solution")

    def test_solve_2d_CPU_vs_GPU(self):  # pylint: disable=invalid-name
        """Compare the solution of the Laplace equation using CPU vs GPU solvers."""
        pot = ScalarField(shape=(32, 32))
        pot.solver = LaplacianSolver()
        pot.add_boundary_condition((0, all), 10)
        pot.add_boundary_condition((-1, all), 5)
        pot.apply_conditions()

        # Solve with GPU
        pot.solver = LaplacianSolverGPU()
        pot.solve_laplace_by_relaxation()
        field_gpu = pot.values.copy()
        pot.show(title=f"{self._testMethodName} [GPU]")

        # Solve with CPU
        pot.values = np.zeros(shape=pot.shape, dtype=np.float32)
        pot.solver = LaplacianSolver()
        pot.solve_laplace_by_relaxation()
        field_cpu = pot.values.copy()

        self.assertTrue((field_cpu - field_gpu).all() == 0)
        pot.show(title=f"{self._testMethodName} [CPU]")

    def test_solve_2d_function_condition(self):
        """Test solving with boundary conditions defined as functions."""
        pot = ScalarField(shape=(32, 32))
        x = np.linspace(0, 6.28, 32)
        pot.add_boundary_condition((0, all), 10 * np.sin(x))
        pot.add_boundary_condition((-1, all), -10 * np.sin(x))
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation()
        pot.show(title=self._testMethodName)

    def test_solve_2d_funky_conditions(self):
        """Test solving with irregularly shaped boundary conditions in 2d."""
        pot = ScalarField(shape=(32, 32))
        pot.add_boundary_condition((0, all), 10)
        pot.add_boundary_condition((-1, all), 5)
        pot.add_boundary_condition((10, slice(10, 20)), 10)
        pot.add_boundary_condition((slice(15, 18), 20), 10)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation()
        pot.show(title=self._testMethodName)

    def test_conditions_3d(self):
        """Test adding and applying boundary conditions in 3d."""
        pot = ScalarField(shape=(32, 32, 32))
        pot.add_boundary_condition((all, 0, 0), 10)
        pot.apply_conditions()

    def test_solve_1d(self):
        """Test solving the Laplace equation in 1d."""
        pot = ScalarField(shape=(32,))
        pot.add_boundary_condition((0,), 10)
        pot.add_boundary_condition((15,), 0)
        pot.add_boundary_condition((-1,), 10)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation(tolerance=1e-8)
        pot.show(title=self._testMethodName)

    def test_solve_3d(self):
        """Test solving the Laplace equation in 3d with boundary conditions."""
        pot = ScalarField(shape=(64, 64, 64))
        pot.add_boundary_condition((-1, all, all), 10)
        pot.add_boundary_condition((all, 0, all), 0)
        pot.add_boundary_condition((all, -1, all), 0)
        pot.add_boundary_condition((all, all, 0), 0)
        pot.add_boundary_condition((all, all, -1), 0)
        pot.add_boundary_condition((0, all, all), 10)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation(tolerance=1e-7)
        pot.show(slices=(all, 31, all), title=self._testMethodName)

    def test_solve_3d_GPU(self):  # pylint: disable=invalid-name
        """Test solving the Laplace equation in 3d using a GPU solver."""
        pot = ScalarField(shape=(32, 32, 32))
        pot.solver = LaplacianSolverGPU()
        pot.add_boundary_condition((-1, all, all), 10)
        pot.add_boundary_condition((all, 0, all), 0)
        pot.add_boundary_condition((all, -1, all), 0)
        pot.add_boundary_condition((all, all, 0), 0)
        pot.add_boundary_condition((all, all, -1), 0)
        pot.add_boundary_condition((0, all, all), 10)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation(tolerance=1e-7)
        pot.show(slices=(all, 31, all), title=self._testMethodName)


if __name__ == "__main__":
    # unittest.main(defaultTest=["PotentialTestCase.test_solve_3d_GPU"])
    unittest.main()
