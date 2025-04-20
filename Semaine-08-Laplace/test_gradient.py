"""
Module for testing the gradient of the ScalarField class and its methods using unittest.


"""

import unittest
import numpy as np
from scalarfield import ScalarField
from utils import all
from vectorfield import VectorField2D, SurfaceDomain
import matplotlib.pyplot as plt


class GradientTestCase(unittest.TestCase):
    """ """

    def test_init(self):
        """Test that a ScalarField instance is initialized properly."""
        self.assertIsNotNone(ScalarField(shape=(32, 32)))

    @unittest.SkipTest
    def test_linear_gradient(self):
        potential = ScalarField(shape=(32, 64))

        gradient_line = np.linspace(0, 1, potential.shape[-1], dtype=np.float32)
        potential.values = np.broadcast_to(gradient_line, potential.shape)
        potential.show()

    @unittest.SkipTest
    def test_field_is_linear_axis_0(self):
        potential = ScalarField(shape=(32, 64))
        potential.set_linear_gradient(potential.shape, axis=0)
        potential.show()

    @unittest.SkipTest
    def test_field_is_linear_axis_1(self):
        potential = ScalarField(shape=(32, 64))
        potential.set_linear_gradient(potential.shape, axis=1)
        potential.show()

    @unittest.SkipTest
    def test_linear_in_x_field_null_gradient_in_x(self):
        """
        A linear field in x gives a constant gradient.
        Field goes from 0 to 1, gradient will be 0.1 in x
        and 0 in y (axis 0 and 1)
        """

        potential = ScalarField(shape=(10, 10))
        potential.set_linear_gradient(potential.shape, axis=0)
        x_grad, y_grad = potential.gradient()

        for v_line in x_grad[1:-1, :]:
            for v in v_line:
                self.assertAlmostEqual(v, 0.1, 4)

        for v_line in y_grad:
            for v in v_line:
                self.assertAlmostEqual(v, 0.0, 4)

        potential.show()

    @unittest.SkipTest
    def test_linear_in_y_field_null_gradient_in_y(self):
        """
        A linear field in y gives a constant gradient.
        Field goes from 0 to 1, gradient will be 0.1 in y
        and 0 in x (axis 0 and 1)
        """

        potential = ScalarField(shape=(10, 10))
        potential.set_linear_gradient(potential.shape, axis=1)
        x_grad, y_grad = potential.gradient()

        for v_line in x_grad[1:-1, :]:
            for v in v_line:
                self.assertAlmostEqual(v, 0.0, 4)

        for v_line in y_grad:
            for v in v_line:
                self.assertAlmostEqual(v, 0.1, 4)

        potential.show()

    @unittest.SkipTest
    def test_use_vectorfield2d_to_visualize(self):
        """
        A linear field in y gives a constant gradient.
        Field goes from 0 to 1, gradient will be 0.1 in y
        and 0 in x (axis 0 and 1)
        """
        N = 20
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)

        surface = SurfaceDomain(X=x, Y=y)

        potential = ScalarField(shape=(N, N))
        potential.set_linear_gradient(potential.shape, axis=0)
        x_grad, y_grad = potential.gradient()
        efield = VectorField2D(surface=surface, U=x_grad, V=y_grad)
        efield.display()

        potential.set_linear_gradient(potential.shape, axis=1)
        x_grad, y_grad = potential.gradient()
        efield = VectorField2D(surface=surface, U=x_grad, V=y_grad)
        efield.display()

    def test_get_value_at_fractional_index_in_array_gradient_x(self):
        potential = ScalarField(shape=(10, 10))

        potential.set_linear_gradient(potential.shape, axis=0)
        for i in np.linspace(0, potential.shape[0] - 1, endpoint=False):
            self.assertAlmostEqual(
                potential.value_at_fractional_index(i, 1),
                potential.value_at_fractional_index(i, 2),
            )

    def test_get_value_at_fractional_index_in_array_gradient_y(self):
        potential = ScalarField(shape=(10, 10))

        potential.set_linear_gradient(potential.shape, axis=1)
        for j in np.linspace(1, potential.shape[1] - 1, endpoint=False):
            self.assertAlmostEqual(
                potential.value_at_fractional_index(1, j),
                potential.value_at_fractional_index(2, j),
            )

    def test_fraction_function(self):
        potential = ScalarField(shape=(10, 10))

        potential.set_linear_gradient(potential.shape, axis=1)
        for j in np.linspace(1, potential.shape[1] - 1, endpoint=False):
            self.assertAlmostEqual(
                value_at_fractional_index(potential.values, 1, j),
                value_at_fractional_index(potential.values, 2, j),
            )

    def test_constant_field_is_constant_acceleration(self):
        """
        A linear field in y gives a constant gradient.
        Field goes from 0 to 1, gradient will be 0.1 in y
        and 0 in x (axis 0 and 1)
        """

        positions = [np.array([0, 0])]
        velocities = [np.array([0, 0])]
        accelerations = [np.array([0, 0])] * 100

        while len(positions) < 100:
            dt = 1
            position = positions[-1]
            velocity = velocities[-1]
            acceleration = np.array([1.1, 0])

            next_velocity = velocity + dt * acceleration
            next_position = position + dt * next_velocity

            positions.append(next_position)
            velocities.append(next_velocity)

        # We obtain a parabola
        # print("\n".join([f"{p[0]:.2f}" for p in positions]))

    def test_get_value_at_fractional_index_in_array_gradient_y(self):
        potential = ScalarField(shape=(10, 10))

        potential.set_linear_gradient(potential.shape, axis=0)

        gradx, grady = potential.gradient()

        positions = [np.array([0, 0])]
        velocities = [np.array([0, 0])]
        accelerations = [np.array([0, 0])] * 100

        while len(positions) < 1000:
            dt = 1
            position = positions[-1]
            velocity = velocities[-1]
            try:
                ax = value_at_fractional_index(gradx * 0.01, position[0], position[1])
                ay = value_at_fractional_index(grady * 0.01, position[0], position[1])
                acceleration = np.array([ax, ay])
            except:
                break

            next_velocity = velocity + dt * acceleration

            possible_next_position = position + dt * next_velocity
            next_position = possible_next_position  # check for collisions later

            positions.append(next_position)
            velocities.append(next_velocity)

        # We obtain a parabola
        # print("\n".join([f"{p[0]:.2f}" for p in positions]))

    def test_collisions(self):
        """
        The strategy to check for a collision is simple in 2D: we create an
        array the same size as the potential array, filled with zeros.
        Then, we put 1 where there is an obstacle. That's the first step.

        The second step is, assuming we go from i0, j0 to i,j, then we
        discretize this in steps of 0.5 (in units of indexes) and step from
        start to finish until we find an obstacle.
        The first point that is not zero is the collision spot.

        """

        obstacles = np.zeros(shape=(10, 10))
        obstacles[4, :] = 1

        start = np.array([0, 3])
        finish = np.array([7, 7])

        points_along_line = []
        for tau in np.linspace(0, 1, 100):
            points_along_line.append(start + tau * (finish - start))

        collision_point = None
        for p in points_along_line:
            try:
                if obstacles[int(p[0]), int(p[1])] != 0:
                    collision_point = p
            except:
                break
        print(collision_point)
        # print(points_along_line)

    def test_collisions_fct(self):
        """
        The strategy to check for a collision is simple in 2D: we create an
        array the same size as the potential array, filled with zeros.
        Then, we put 1 where there is an obstacle. That's the first step.

        The second step is, assuming we go from i0, j0 to i,j, then we
        discretize this in steps of 0.5 (in units of indexes) and step from
        start to finish until we find an obstacle.
        The first point that is not zero is the collision spot.

        """

        obstacles = np.zeros(shape=(10, 10))
        obstacles[4, :] = 1

        start = np.array([0, 3])
        end = np.array([7, 7])

        print(is_colliding(obstacles, start, end))
        # print(points_along_line)

    def test_plot_trajectory(self):
        pts = [np.array([0, 0]), np.array([1, 1]), np.array([1, 2]), np.array([2, 1.5])]
        coords = np.stack(pts)
        x, y = coords[:, 0], coords[:, 1]
        plt.plot(x, y)
        plt.show()


def is_colliding(obstacles, start, end):
    collision_point = None

    points_along_line = []
    for tau in np.linspace(0, 1, 100):
        points_along_line.append(start + tau * (end - start))

    for p in points_along_line:
        try:
            if obstacles[int(p[0]), int(p[1])] != 0:
                collision_point = p
        except:
            break

    return collision_point


def value_at_fractional_index(array, i_float: float, j_float: float):
    """
    We will often need to value in between discrete steps.
    We will take a linear interpolation beteen the two values
    """

    if i_float < 0 or i_float >= array.shape[0] - 1:
        raise ValueError(f"Outside of the range in i : {i_float}")
    if j_float < 0 or j_float > array.shape[1] - 1:
        raise ValueError(f"Outside of the range in j : {j_float}")

    i = int(np.floor_divide(i_float, 1))
    i_frac = np.remainder(i_float, 1)
    j = int(np.floor_divide(j_float, 1))
    j_frac = np.remainder(j_float, 1)

    return (
        array[i, j]
        + i_frac * (array[i + 1, j] - array[i, j])
        + j_frac * (array[i, j + 1] - array[i, j])
    )


if __name__ == "__main__":
    unittest.main()
