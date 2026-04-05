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


class TestScalarFieldMethods(unittest.TestCase):
    def test_shape(self):
        sf = ScalarField(shape=(10, 20))
        self.assertEqual(sf.shape, (10, 20))

    def test_reset(self):
        sf = ScalarField(shape=(10, 10))
        sf.values[5, 5] = 42
        sf.reset()
        self.assertTrue(np.all(sf.values == 0))

    def test_reset_new_shape(self):
        sf = ScalarField(shape=(10, 10))
        sf.reset(shape=(20, 20))
        self.assertEqual(sf.shape, (20, 20))

    def test_calibration_default(self):
        sf = ScalarField(shape=(10, 10))
        self.assertEqual(sf.calibration, [1, 1])

    def test_calibration_custom(self):
        sf = ScalarField(shape=(10, 10), calibration=[0.5, 0.5])
        self.assertEqual(sf.calibration, [0.5, 0.5])

    def test_gradient_uniform_field(self):
        """Gradient of a uniform field should be zero everywhere."""
        sf = ScalarField(shape=(20, 20))
        sf.values[:] = 5.0
        gx, gy = sf.gradient()
        np.testing.assert_allclose(gx, 0, atol=1e-6)
        np.testing.assert_allclose(gy, 0, atol=1e-6)

    def test_gradient_linear_axis0(self):
        """Gradient of a linear ramp along axis 0 should be constant."""
        sf = ScalarField(shape=(20, 20))
        for i in range(20):
            sf.values[i, :] = float(i)
        gx, gy = sf.gradient()
        np.testing.assert_allclose(gx[1:-1, :], 1.0, atol=1e-6)
        np.testing.assert_allclose(gy[:, 1:-1], 0.0, atol=1e-6)

    def test_gradient_linear_axis1(self):
        """Gradient of a linear ramp along axis 1 should be constant."""
        sf = ScalarField(shape=(20, 20))
        for j in range(20):
            sf.values[:, j] = float(j)
        gx, gy = sf.gradient()
        np.testing.assert_allclose(gx[1:-1, :], 0.0, atol=1e-6)
        np.testing.assert_allclose(gy[:, 1:-1], 1.0, atol=1e-6)

    def test_set_linear_gradient_axis0(self):
        sf = ScalarField(shape=(10, 10))
        sf.set_linear_gradient((10, 10), axis=0)
        self.assertGreater(sf.values[-1, 0], sf.values[0, 0])
        np.testing.assert_allclose(sf.values[:, 0], sf.values[:, 5])

    def test_set_linear_gradient_axis1(self):
        sf = ScalarField(shape=(10, 10))
        sf.set_linear_gradient((10, 10), axis=1)
        self.assertGreater(sf.values[0, -1], sf.values[0, 0])
        np.testing.assert_allclose(sf.values[0, :], sf.values[5, :])

    def test_boundary_condition_stored(self):
        sf = ScalarField(shape=(10, 10))
        sf.add_boundary_condition((0, all), 5.0)
        self.assertEqual(len(sf.conditions), 1)

    def test_apply_conditions_sets_values(self):
        sf = ScalarField(shape=(10, 10))
        sf.add_boundary_condition((0, all), 7.0)
        sf.apply_conditions()
        np.testing.assert_allclose(sf.values[0, :], 7.0)

    def test_apply_conditions_function(self):
        sf = ScalarField(shape=(10, 10))
        sf.add_boundary_function(lambda v: v.__setitem__((0, all), 3.0))
        sf.apply_conditions()
        np.testing.assert_allclose(sf.values[0, :], 3.0)

    def test_boundary_mask(self):
        sf = ScalarField(shape=(20, 20))
        sf.add_boundary_condition((slice(0, 5), slice(0, 5)), 1.0)
        sf.add_boundary_condition((slice(10, 15), slice(10, 15)), 2.0)
        mask = sf.boundary_mask
        self.assertTrue(np.all(mask[0:5, 0:5]))
        self.assertTrue(np.all(mask[10:15, 10:15]))
        self.assertFalse(mask[7, 7])

    def test_boundary_outline_outside_mask(self):
        sf = ScalarField(shape=(20, 20))
        sf.add_boundary_condition((slice(5, 10), slice(5, 10)), 1.0)
        outline, nx, ny = sf.boundary_outline()
        mask = sf.boundary_mask
        self.assertFalse(np.any(outline & mask))
        self.assertTrue(np.any(outline))

    def test_boundary_outline_normals_unit_length(self):
        sf = ScalarField(shape=(20, 20))
        sf.add_boundary_condition((slice(5, 10), slice(5, 10)), 1.0)
        outline, nx, ny = sf.boundary_outline()
        norm = np.sqrt(nx[outline]**2 + ny[outline]**2)
        np.testing.assert_allclose(norm, 1.0, atol=0.01)

    def test_value_at_fractional_index_on_grid(self):
        sf = ScalarField(shape=(10, 10))
        sf.values[3, 4] = 7.0
        self.assertAlmostEqual(sf.value_at_fractional_index(3.0, 4.0), 7.0)

    def test_value_at_fractional_index_interpolates(self):
        sf = ScalarField(shape=(10, 10))
        sf.values[2, 2] = 0.0
        sf.values[3, 2] = 10.0
        val = sf.value_at_fractional_index(2.5, 2.0)
        self.assertAlmostEqual(val, 5.0, places=3)

    def test_value_at_fractional_index_out_of_range(self):
        sf = ScalarField(shape=(10, 10))
        with self.assertRaises(ValueError):
            sf.value_at_fractional_index(-1, 0)
        with self.assertRaises(ValueError):
            sf.value_at_fractional_index(0, 10)

    def test_upscale(self):
        sf = ScalarField(shape=(10, 10))
        sf.upscale(factor=2)
        self.assertEqual(sf.shape, (20, 20))

    def test_solve_parallel_plates(self):
        """Two parallel plates: solution should be a linear gradient."""
        sf = ScalarField(shape=(50, 50))
        expected_row = np.linspace(10.0, 0.0, 50, dtype=np.float32)
        sf.add_boundary_condition((all, 0), 10.0)
        sf.add_boundary_condition((all, -1), 0.0)
        sf.add_boundary_condition((0, all), expected_row)
        sf.add_boundary_condition((-1, all), expected_row)
        sf.apply_conditions()
        sf.solve_laplace_by_relaxation()
        mid_row = sf.values[25, :]
        np.testing.assert_allclose(mid_row, expected_row, atol=0.1)

    def test_save_and_load(self):
        import tempfile, os
        sf = ScalarField(shape=(10, 10))
        sf.values[5, 5] = 42.0
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = f.name
        try:
            sf.save(path)
            loaded = np.load(path)
            np.testing.assert_array_equal(sf.values, loaded)
        finally:
            os.unlink(path)


class TestScalarFieldDemo(unittest.TestCase):
    """Tests visuels pour demonstrer ScalarField a quelqu'un qui ne l'a jamais utilise."""

    def test_demo_01_deux_plaques_paralleles(self):
        """Le cas le plus simple: deux plaques infinies a potentiel different.
        Plaque gauche a 10V, plaque droite a 0V.
        On impose un gradient lineaire sur les bords haut et bas pour
        simuler des plaques infinies. La solution est un gradient lineaire."""
        import matplotlib.pyplot as plt

        sf = ScalarField(shape=(500, 100))
        sf.add_boundary_condition((all, 0), 10.0)    # plaque gauche a 10V
        sf.add_boundary_condition((all, -1), 0.0)    # plaque droite a 0V
        sf.apply_conditions()
        sf.solve_laplace_by_relaxation()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(sf.values, origin='lower', cmap='RdBu_r', aspect='auto')
        axes[0].set_title("Potentiel: gauche=10V, droite=0V")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(axes[0].images[0], ax=axes[0], label="V")

        axes[1].plot(sf.values[250, :])
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("V")
        axes[1].set_title("Profil a mi-hauteur (lineaire)")
        axes[1].grid(True)
        plt.tight_layout()
        plt.pause(0.5)
        plt.close()

    def test_demo_02_boite_avec_conditions(self):
        """Quatre cotes avec des potentiels differents:
        haut=10V, bas=0V, gauche=droite=5V.
        On voit comment le potentiel s'adapte partout a l'interieur."""
        import matplotlib.pyplot as plt

        sf = ScalarField(shape=(100, 100))
        sf.add_boundary_condition((0, all), 10.0)    # haut
        sf.add_boundary_condition((-1, all), 0.0)    # bas
        sf.add_boundary_condition((all, 0), 5.0)     # gauche
        sf.add_boundary_condition((all, -1), 5.0)    # droite
        sf.apply_conditions()
        sf.solve_laplace_by_relaxation()

        plt.figure(figsize=(6, 5))
        plt.imshow(sf.values, origin='lower', cmap='RdBu_r')
        plt.colorbar(label="V")
        plt.title("Boite: haut=10V, bas=0V, cotes=5V")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(0.5)
        plt.close()

    def test_demo_03_condition_sinusoidale(self):
        """Condition frontiere sinusoidale en haut (V = 10*sin(x)),
        les trois autres cotes a 0V.
        Le potentiel oscille et s'attenue vers le bas."""
        import matplotlib.pyplot as plt

        sf = ScalarField(shape=(100, 100))
        x = np.linspace(0, 2 * np.pi, 100)
        sf.add_boundary_condition((0, all), 10 * np.sin(x).astype(np.float32))  # haut: 10*sin(x)
        sf.add_boundary_condition((-1, all), 0.0)   # bas: 0V
        sf.add_boundary_condition((all, 0), 0.0)     # gauche: 0V
        sf.add_boundary_condition((all, -1), 0.0)    # droite: 0V
        sf.apply_conditions()
        sf.solve_laplace_by_relaxation()

        plt.figure(figsize=(6, 5))
        plt.imshow(sf.values, origin='lower', cmap='RdBu_r')
        plt.colorbar(label="V")
        plt.title("Condition sinusoidale en haut")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(0.5)
        plt.close()

    def test_demo_04_conducteur_central(self):
        """Un conducteur carre au centre a 10V, entoure de murs a 0V.
        Les lignes de potentiel rayonnent du conducteur."""
        import matplotlib.pyplot as plt

        sf = ScalarField(shape=(100, 100))
        sf.add_boundary_condition((0, all), 0.0)       # haut: 0V
        sf.add_boundary_condition((-1, all), 0.0)      # bas: 0V
        sf.add_boundary_condition((all, 0), 0.0)       # gauche: 0V
        sf.add_boundary_condition((all, -1), 0.0)      # droite: 0V
        sf.add_boundary_condition((slice(40, 60), slice(40, 60)), 10.0)  # carre central: 10V
        sf.apply_conditions()
        sf.solve_laplace_by_relaxation()

        plt.figure(figsize=(6, 5))
        plt.imshow(sf.values, origin='lower', cmap='hot')
        plt.colorbar(label="V")
        plt.contour(sf.values, levels=10, colors='white', linewidths=0.5)
        plt.title("Conducteur central=10V, murs=0V")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(0.5)
        plt.close()

    def test_demo_05_gradient_et_champ_electrique(self):
        """On resout le potentiel puis on calcule le champ electrique E = -grad(V).
        Les fleches montrent la direction et l'intensite du champ."""
        import matplotlib.pyplot as plt

        sf = ScalarField(shape=(80, 80))
        sf.add_boundary_condition((0, all), 0.0)
        sf.add_boundary_condition((-1, all), 0.0)
        sf.add_boundary_condition((all, 0), 0.0)
        sf.add_boundary_condition((all, -1), 0.0)
        sf.add_boundary_condition((slice(30, 50), slice(30, 50)), 10.0)
        sf.apply_conditions()
        sf.solve_laplace_by_relaxation()

        Ex, Ey = sf.gradient()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].imshow(sf.values, origin='lower', cmap='hot')
        axes[0].set_title("Potentiel V")
        fig.colorbar(axes[0].images[0], ax=axes[0], label="V")

        s = slice(None, None, 4)  # une fleche sur 4
        X, Y = np.meshgrid(np.arange(80), np.arange(80))
        norm = np.sqrt(Ex**2 + Ey**2)
        axes[1].imshow(norm, origin='lower', cmap='viridis', alpha=0.5)
        axes[1].quiver(X[s, s], Y[s, s], -Ey[s, s], -Ex[s, s], color='red')
        axes[1].set_title("Champ electrique E = -grad(V)")
        fig.colorbar(axes[1].images[0], ax=axes[1], label="|E|")

        plt.tight_layout()
        plt.pause(0.5)
        plt.close()

    def test_demo_06_outline_et_normales(self):
        """On montre le contour (outline) d'un conducteur et les normales sortantes.
        Utile pour calculer la charge de surface."""
        import matplotlib.pyplot as plt

        sf = ScalarField(shape=(80, 80))
        sf.add_boundary_condition((0, all), 0.0)
        sf.add_boundary_condition((-1, all), 0.0)
        sf.add_boundary_condition((all, 0), 0.0)
        sf.add_boundary_condition((all, -1), 0.0)
        sf.add_boundary_condition((slice(25, 55), slice(25, 55)), 10.0)
        sf.apply_conditions()
        sf.solve_laplace_by_relaxation()

        mask = np.zeros(sf.shape, dtype=bool)
        mask[25:55, 25:55] = True
        outline, nx, ny = sf.boundary_outline(mask)
        oy, ox = np.where(outline)

        plt.figure(figsize=(6, 6))
        plt.imshow(sf.values, origin='lower', cmap='hot')
        plt.colorbar(label="V")
        plt.quiver(ox, oy, ny[outline], nx[outline], color='cyan', scale=20)
        plt.title("Outline du conducteur avec normales sortantes")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(0.5)
        plt.close()

    def test_demo_07_densite_de_charge_surface(self):
        """On calcule sigma = epsilon_0 * E·n a la surface d'un conducteur.
        Par la loi de Gauss: epsilon_0 * E_n = sigma (densite de charge surfacique).
        La densite de charge est plus elevee aux coins."""
        import matplotlib.pyplot as plt

        sf = ScalarField(shape=(100, 100))
        sf.add_boundary_condition((0, all), 0.0)
        sf.add_boundary_condition((-1, all), 0.0)
        sf.add_boundary_condition((all, 0), 0.0)
        sf.add_boundary_condition((all, -1), 0.0)
        sf.add_boundary_condition((slice(35, 65), slice(35, 65)), 10.0)
        sf.apply_conditions()
        sf.solve_laplace_by_relaxation()

        Ex, Ey = sf.gradient()

        mask = np.zeros(sf.shape, dtype=bool)
        mask[35:65, 35:65] = True
        outline, nx, ny = sf.boundary_outline(mask)
        oy, ox = np.where(outline)

        # sigma = epsilon_0 * E·n = -grad(V)·n  (epsilon_0 = 1 en unites de grille)
        sigma = -(Ex[outline] * nx[outline] + Ey[outline] * ny[outline])

        plt.figure(figsize=(6, 6))
        plt.scatter(ox, oy, c=sigma, cmap='coolwarm', s=10)
        plt.colorbar(label=r"$\sigma = \epsilon_0 \, \mathbf{E} \cdot \hat{n}$")
        plt.title(r"$\epsilon_0 \, E_n = \sigma$ (loi de Gauss)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        plt.pause(0.5)
        plt.close()


    def test_demo_08_boundary_mask(self):
        """boundary_mask retourne un masque booleen de tous les pixels
        qui ont une condition frontiere. Utile pour visualiser la geometrie
        avant de resoudre."""
        import matplotlib.pyplot as plt

        sf = ScalarField(shape=(100, 100))
        sf.add_boundary_condition((0, all), 0.0)                           # haut: 0V
        sf.add_boundary_condition((-1, all), 0.0)                          # bas: 0V
        sf.add_boundary_condition((all, 0), 10.0)                          # gauche: 10V
        sf.add_boundary_condition((all, -1), 0.0)                          # droite: 0V
        sf.add_boundary_condition((slice(40, 60), slice(40, 60)), 5.0)     # carre central: 5V
        sf.add_boundary_condition((slice(20, 25), slice(70, 90)), 8.0)     # rectangle: 8V

        mask = sf.boundary_mask

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(mask.astype(float), origin='lower', cmap='gray')
        axes[0].set_title("boundary_mask: pixels avec condition frontiere")

        sf.apply_conditions()
        axes[1].imshow(sf.values, origin='lower', cmap='RdBu_r')
        axes[1].set_title("Valeurs apres apply_conditions (avant resolution)")
        fig.colorbar(axes[1].images[0], ax=axes[1], label="V")

        plt.tight_layout()
        plt.pause(0.5)
        plt.close()

    def test_demo_09_outline_simple(self):
        """boundary_outline retourne les pixels juste a l'exterieur
        d'une region. On peut l'utiliser avec le masque complet (toutes
        les conditions) ou avec un masque specifique."""
        import matplotlib.pyplot as plt

        sf = ScalarField(shape=(100, 100))
        sf.add_boundary_condition((0, all), 0.0)
        sf.add_boundary_condition((-1, all), 0.0)
        sf.add_boundary_condition((all, 0), 0.0)
        sf.add_boundary_condition((all, -1), 0.0)
        sf.add_boundary_condition((slice(30, 50), slice(20, 40)), 10.0)   # rectangle A
        sf.add_boundary_condition((slice(60, 80), slice(50, 80)), 5.0)    # rectangle B

        # Outline de toutes les conditions
        outline_all, _, _ = sf.boundary_outline()

        # Outline du rectangle A seulement
        mask_a = np.zeros(sf.shape, dtype=bool)
        mask_a[30:50, 20:40] = True
        outline_a, _, _ = sf.boundary_outline(mask_a)

        # Outline du rectangle B seulement
        mask_b = np.zeros(sf.shape, dtype=bool)
        mask_b[60:80, 50:80] = True
        outline_b, _, _ = sf.boundary_outline(mask_b)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].imshow(outline_all.astype(float), origin='lower', cmap='gray')
        axes[0].set_title("boundary_outline() — tout")

        axes[1].imshow(sf.boundary_mask.astype(float), origin='lower', cmap='gray', alpha=0.3)
        axes[1].imshow(outline_a.astype(float), origin='lower', cmap='Reds', alpha=0.7)
        axes[1].set_title("Outline rectangle A (10V)")

        axes[2].imshow(sf.boundary_mask.astype(float), origin='lower', cmap='gray', alpha=0.3)
        axes[2].imshow(outline_b.astype(float), origin='lower', cmap='Blues', alpha=0.7)
        axes[2].set_title("Outline rectangle B (5V)")

        plt.tight_layout()
        plt.pause(0.5)
        plt.close()

    def test_demo_10_normales_sur_formes(self):
        """Les normales sortantes permettent de calculer le flux E·n.
        On montre les normales sur deux conducteurs de formes differentes."""
        import matplotlib.pyplot as plt

        sf = ScalarField(shape=(100, 100))
        sf.add_boundary_condition((0, all), 0.0)
        sf.add_boundary_condition((-1, all), 0.0)
        sf.add_boundary_condition((all, 0), 0.0)
        sf.add_boundary_condition((all, -1), 0.0)

        # Rectangle horizontal
        sf.add_boundary_condition((slice(45, 55), slice(10, 45)), 10.0)
        # Petit carre
        sf.add_boundary_condition((slice(30, 45), slice(60, 75)), -5.0)

        sf.apply_conditions()
        sf.solve_laplace_by_relaxation()

        mask_rect = np.zeros(sf.shape, dtype=bool)
        mask_rect[45:55, 10:45] = True
        outline_r, nx_r, ny_r = sf.boundary_outline(mask_rect)
        oy_r, ox_r = np.where(outline_r)

        mask_sq = np.zeros(sf.shape, dtype=bool)
        mask_sq[30:45, 60:75] = True
        outline_s, nx_s, ny_s = sf.boundary_outline(mask_sq)
        oy_s, ox_s = np.where(outline_s)

        plt.figure(figsize=(8, 7))
        plt.imshow(sf.values, origin='lower', cmap='RdBu_r')
        plt.colorbar(label="V")
        # Dessiner les normales comme des segments
        length = 3
        s = slice(None, None, 2)
        for ox, oy, nx_vals, ny_vals, color in [
            (ox_r, oy_r, nx_r[outline_r], ny_r[outline_r], 'black'),
            (ox_s, oy_s, nx_s[outline_s], ny_s[outline_s], 'green'),
        ]:
            for x, y, dnx, dny in zip(ox[s], oy[s], ny_vals[s], nx_vals[s]):
                plt.plot([x, x + length * dnx], [y, y + length * dny], color=color, lw=0.8)
        plt.title("Normales sortantes: rectangle (10V) et carre (-5V)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(0.5)
        plt.close()

    def test_demo_11_flux_et_charge_totale(self):
        """On calcule la charge totale Q = sum(epsilon_0 * E·n) sur le
        contour d'un conducteur. Par la loi de Gauss, Q_pos + Q_neg = 0."""
        import matplotlib.pyplot as plt

        sf = ScalarField(shape=(100, 100))
        sf.add_boundary_condition((0, all), 0.0)       # murs a 0V
        sf.add_boundary_condition((-1, all), 0.0)
        sf.add_boundary_condition((all, 0), 0.0)
        sf.add_boundary_condition((all, -1), 0.0)
        sf.add_boundary_condition((slice(20, 40), slice(20, 40)), 10.0)   # conducteur A: 10V
        sf.add_boundary_condition((slice(60, 80), slice(60, 80)), -10.0)  # conducteur B: -10V
        sf.apply_conditions()
        sf.solve_laplace_by_relaxation()

        Ex, Ey = sf.gradient()

        mask_a = np.zeros(sf.shape, dtype=bool)
        mask_a[20:40, 20:40] = True
        outline_a, nx_a, ny_a = sf.boundary_outline(mask_a)
        Q_a = np.sum(-(Ex[outline_a] * nx_a[outline_a] + Ey[outline_a] * ny_a[outline_a]))

        mask_b = np.zeros(sf.shape, dtype=bool)
        mask_b[60:80, 60:80] = True
        outline_b, nx_b, ny_b = sf.boundary_outline(mask_b)
        Q_b = np.sum(-(Ex[outline_b] * nx_b[outline_b] + Ey[outline_b] * ny_b[outline_b]))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].imshow(sf.values, origin='lower', cmap='RdBu_r')
        axes[0].set_title("Potentiel: A=10V, B=-10V, murs=0V")
        fig.colorbar(axes[0].images[0], ax=axes[0], label="V")

        oy_a, ox_a = np.where(outline_a)
        oy_b, ox_b = np.where(outline_b)
        sigma_a = -(Ex[outline_a] * nx_a[outline_a] + Ey[outline_a] * ny_a[outline_a])
        sigma_b = -(Ex[outline_b] * nx_b[outline_b] + Ey[outline_b] * ny_b[outline_b])

        axes[1].scatter(ox_a, oy_a, c=sigma_a, cmap='coolwarm', s=8, vmin=-2, vmax=2)
        axes[1].scatter(ox_b, oy_b, c=sigma_b, cmap='coolwarm', s=8, vmin=-2, vmax=2)
        axes[1].set_title(f"Q(A) = {Q_a:.2f},  Q(B) = {Q_b:.2f}")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].axis('equal')

        plt.tight_layout()
        plt.pause(0.5)
        plt.close()


if __name__ == "__main__":
    unittest.main()
