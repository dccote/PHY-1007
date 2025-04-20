"""
A class ScalarField to represent a scalar field in 2D or 3D.
If the field obeys the Laplace equation, it can be solved
and visualised.

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from solvers import LaplacianSolver
from utils import left, center, right


class ScalarField:
    """
    Represents a scalar field with methods for boundary conditions,
    Laplacian solving, and visualization.
    """

    def __init__(self, shape, calibration=None):
        """
        Initializes the scalar field with a given shape (2D or 3D).

        Parameters:
        shape (tuple): The shape of the field (2D or 3D array expected).
        """
        self.values = np.zeros(shape=shape, dtype=np.float32)
        if calibration is None:
            calibration = [1] * len(shape)

        self.calibration = calibration

        self.conditions = []
        self.condition_fct = None
        self.solver = LaplacianSolver()

    def save(self, filepath):
        self.array.save(filepath)

    @property
    def shape(self):
        """
        Convenience function to return the shape of the scalar field.

        Returns:
        tuple: The shape of the field.
        """
        return self.values.shape

    def set_linear_gradient(self, shape, axis):
        if axis == 0:
            grad_line = np.linspace(0, 1, shape[0], endpoint=False, dtype=np.float32)
            self.values = np.tile(grad_line[:, np.newaxis], (1, shape[1]))
        elif axis == 1:
            grad_line = np.linspace(0, 1, shape[1], endpoint=False, dtype=np.float32)
            self.values = np.tile(grad_line[np.newaxis, :], (shape[0], 1))

    def gradient(self):
        """
        We take the value shifted to the left and to the right to set the
        value at a given point. However, this means we do not know the values
        at the edges because they do not have a neighbour on one side.

        We aren't sending people to space, or running a hospital, therefore
        the values on the edge will be calculated from a single neighbour.

        """

        # First axis (0)
        gradient_x = np.zeros(self.values.shape)
        gradient_x[1:-1, :] = (self.values[right, :] - self.values[left, :]) / 2

        # Right edge
        gradient_x[0, :] = self.values[1, :] - self.values[0, :]
        # Left edge
        gradient_x[-1, :] = self.values[-1, :] - self.values[-2, :]

        gradient_y = np.zeros(self.values.shape)
        gradient_y[:, 1:-1] = (self.values[:, right] - self.values[:, left]) / 2
        # Top edge
        gradient_y[:, 0] = self.values[:, 1] - self.values[:, 0]
        # bottom edge
        gradient_y[:, -1] = self.values[:, -1] - self.values[:, -2]
        return (gradient_x, gradient_y)

    def reset(self, shape=None):
        """
        Resets the scalar field to zeros with the given shape (if provided)

        Parameters:
        shape (tuple, optional): The new shape of the field. Defaults to the current shape.
        """
        if shape is None:
            shape = self.values.shape

        self.values = np.zeros(shape=shape, dtype=np.float32)

    def upscale(self, factor=8, order=2):
        """
        Upscales the scalar field using interpolation: this is useful
        when solving the Laplace equation, because a lower resoltuion
        solution can be obtained, upscaled, and used as the starting
        point at a higher resolution. The defaults are reasonable.

        Parameters:
        factor (int, optional): The scaling factor. Defaults to 8.
        order (int, optional): The interpolation order. Defaults to 2.
        """
        self.values = zoom(self.values, factor, order=order)

    def add_boundary_condition(self, index_or_slices, value_or_values):
        """
        Adds a boundary condition to the scalar field. This is used after each
        iteration of the relaxation method, because points that are fixed (i.e.
        bounadray points) get modified during the iteration, but they need
        to be added back for the following iteration.

        Parameters:
        index_or_slices (tuple or slice): Indices or slices where the condition applies.
        value_or_values (float or np.ndarray): The value(s) to assign.
        """
        self.conditions.append((index_or_slices, value_or_values))

    def add_boundary_function(self, fct):
        """
        Adds a boundary condition function to the scalar field. It must take the values as an input
        and modify the values accordingly.  This is used after each
        iteration of the relaxation method, because points that are fixed (i.e.
        bounadray points) get modified during the iteration, but they need
        to be added back for the following iteration.

        Parameters:
        fct: function that takes a numpy array and modifies it with boundary conditions
        """
        self.condition_fct = fct

    def apply_conditions(self):
        """
        Applies all stored boundary conditions to the scalar field.
        """
        if self.condition_fct is None:
            for index_or_slices, value in self.conditions:
                self.values[*index_or_slices] = value
        else:
            self.condition_fct(self.values)

    def solve_laplace_by_relaxation_with_refinements(
        self, factors=None, tolerance=1e-7
    ):
        """
        Solves the Laplace equation using relaxation with multi-scale refinements.

        Parameters:
        factors (list, optional): Scaling factors for multi-resolution refinement.
                                  Computed automatically if None.
        tolerance (float, optional): Convergence tolerance. Defaults to 1e-7.

        Returns:
        ScalarField: The updated scalar field after solving.
        """
        final_shape = self.shape

        if factors is None:
            factors = []
            shape = self.shape[0]
            while shape > 16:
                factors.append(8)
                shape /= 8

        if factors == []:
            factors = [1]
        else:
            total_factor = math.prod(factors)
            new_shape = np.array(self.shape) // total_factor
            if np.min(new_shape) < 3:
                raise ValueError(
                    f"The total scaling factor {total_factor} from {factors} is too large, the resulting image is too small."
                )

            self.reset(new_shape)

        print(
            f"Requiring {final_shape}, scaling by {factors}, starting with: {self.shape}"
        )
        while True:
            self.solve_laplace_by_relaxation(tolerance=tolerance)

            if len(factors) != 0:
                f = factors.pop()
                self.upscale(factor=f, order=2)
            else:
                break

        return self

    def solve_laplace_by_relaxation(self, tolerance=1e-7):
        """
        Solves the Laplace equation using the relaxation method

        Parameters:
        tolerance (float, optional): Convergence tolerance. Defaults to 1e-7.
        """
        self.solver.solve_by_relaxation(self, tolerance=tolerance)

    def show(self, slices=None, title=None, block=False):
        """
        Displays the scalar field using Matplotlib.

        Parameters:
        slices (tuple, optional): Indices for slicing a subset of the field.
        title (str, optional): Title of the plot.
        block (bool, optional): Whether to block execution until the plot is closed.
                                Defaults to False.
        """
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
