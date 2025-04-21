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
    and visualization. Tests are performed for 1D, 2D, and 3D fields,
    with comparisons between CPU and GPU solvers.
    """

    def test_init(self):
        """Test that a ScalarField instance is initialized properly."""
        self.assertIsNotNone(ScalarField(shape=(32, 32)))

    def test_show(self):
        """Test the visualization of an empty ScalarField."""
        ScalarField(shape=(32, 32)).show(title=self._testMethodName)

    def test_conditions_2d(self):
        """Test adding a simple boundary condition in 2D."""
        pot = ScalarField(shape=(32, 32))
        pot.add_boundary_condition((all, 0), 10)

    def test_apply_conditions_2d(self):
        """Test applying boundary conditions in 2D."""
        pot = ScalarField(shape=(32, 32))
        pot.add_boundary_condition((0, all), 10)
        pot.add_boundary_condition((-1, all), 5)
        pot.apply_conditions()
        pot.show(title=self._testMethodName)

    def test_solve_2d(self):
        """Test solving the Laplace equation in 2D using relaxation."""
        pot = ScalarField(shape=(32, 32))
        pot.add_boundary_condition((0, all), 10)
        pot.add_boundary_condition((-1, all), 5)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation()
        pot.show(title=self._testMethodName)

    def test_solve_2d_then_upscale(self):
        """Test solving the Laplace equation in 2D, then upscaling the solution and refining it via further relaxation."""
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
        """Compare the solution of the Laplace equation using CPU and GPU solvers for numerical consistency."""
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
        """Test solving the Laplace equation with boundary conditions defined as functions in 2D."""
        pot = ScalarField(shape=(32, 32))
        x = np.linspace(0, 6.28, 32)
        pot.add_boundary_condition((0, all), 10 * np.sin(x))
        pot.add_boundary_condition((-1, all), -10 * np.sin(x))
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation()
        pot.show(title=self._testMethodName)

    def test_solve_2d_funky_conditions(self):
        """Test solving the Laplace equation with non-rectangular boundary conditions in 2D."""
        pot = ScalarField(shape=(32, 32))
        pot.add_boundary_condition((0, all), 10)
        pot.add_boundary_condition((-1, all), 5)
        pot.add_boundary_condition((10, slice(10, 20)), 10)
        pot.add_boundary_condition((slice(15, 18), 20), 10)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation()
        pot.show(title=self._testMethodName)

    def test_conditions_3d(self):
        """Test adding and applying boundary conditions in 3D."""
        pot = ScalarField(shape=(32, 32, 32))
        pot.add_boundary_condition((all, 0, 0), 10)
        pot.apply_conditions()

    def boundaries(self, array):
        """Boundary function that sets a central high-potential point in 2D or 3D arrays."""
        if array.ndim == 3:
            a, b, c = array.shape
            array[a // 2, b // 2, c // 2] = 100
        else:
            a, b = array.shape
            array[a // 2, b // 2] = 100

    def test_conditions_2d_fct(self):
        """Test adding and applying boundary conditions via a function in 2D."""
        pot = ScalarField(shape=(32, 32))
        pot.add_boundary_function(self.boundaries)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation(tolerance=1e-7)
        pot.show()

    def test_conditions_2d_fct_with_refinement(self):
        """Test adding boundary conditions via a function in 2D, with multi-scale refinement."""
        pot = ScalarField(shape=(128, 128))
        pot.add_boundary_function(self.boundaries)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation_with_refinements([8, 4], tolerance=1e-7)
        pot.show()

    def test_solve_2d_with_refinement_too_high(self):
        """Test that using too large a refinement scale raises an error in 2D relaxation."""
        pot = ScalarField(shape=(128, 128))
        pot.add_boundary_function(self.boundaries)
        pot.apply_conditions()
        with self.assertRaises(ValueError):
            pot.solve_laplace_by_relaxation_with_refinements([64], tolerance=1e-7)

    def test_conditions_3d_fct(self):
        """Test adding and applying boundary conditions via a function in 3D."""
        pot = ScalarField(shape=(32, 32, 32))
        pot.add_boundary_function(self.boundaries)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation(tolerance=1e-7)

    def test_solve_1d(self):
        """Test solving the Laplace equation in 1D."""
        pot = ScalarField(shape=(32,))
        pot.add_boundary_condition((0,), 10)
        pot.add_boundary_condition((15,), 0)
        pot.add_boundary_condition((-1,), 10)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation(tolerance=1e-8)
        pot.show(title=self._testMethodName)

    def test_solve_3d(self):
        """Test solving the Laplace equation in 3D with boundary conditions."""
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
        """Test solving the Laplace equation in 3D using a GPU solver."""
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

    def test_conditions_2d_save(self):
        """Test adding boundary conditions via a function in 2D, with multi-scale refinement."""
        pot = ScalarField(shape=(128, 128))
        pot.add_boundary_function(self.boundaries)
        pot.apply_conditions()
        pot.solve_laplace_by_relaxation_with_refinements([8, 4], tolerance=1e-7)
        pot.save("potential.npy")


if __name__ == "__main__":
    # unittest.main(defaultTest=["PotentialTestCase.test_conditions_2d_fct_with_refinement"])
    unittest.main()
