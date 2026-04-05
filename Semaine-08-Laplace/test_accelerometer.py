import unittest
import numpy as np
import matplotlib.pyplot as plt
from scalarfield import ScalarField


def interdigitated_potential(S, N, l, t, d=0, gap=None, V_pos=3, V_neg=0):
    """
    Build a ScalarField for an interdigitated capacitor and solve Laplace.

    Parameters
    ----------
    S : int - grid size (S x S)
    N : int - number of finger pairs
    l : int - finger length in pixels
    t : int - finger thickness in pixels
    d : int - lateral offset of negative fingers (0 = perfectly aligned)
    gap : int or None - gap between fingers. If None, uses delta = S // N.
    V_pos : float - voltage on positive side
    V_neg : float - voltage on negative side

    Returns
    -------
    potential : ScalarField (solved)
    """
    if gap is None:
        delta = S // N
    else:
        delta = t + gap

    offset = delta // 2 + d
    assert offset > 1

    potential = ScalarField(shape=(S, S))

    potential.add_boundary_condition((slice(0, S), slice(0, 3)), V_pos)
    potential.add_boundary_condition((slice(0, S), slice(S - 3, S)), V_neg)

    for i in range(N):
        finger = (slice(i * delta, i * delta + t), slice(0, l))
        potential.add_boundary_condition(finger, V_pos)

    for i in range(N):
        finger = (slice(offset + i * delta, offset + i * delta + t), slice(S - l, S))
        potential.add_boundary_condition(finger, V_neg)

    potential.apply_conditions()
    potential.solve_laplace_by_relaxation()
    return potential


def interdigitated_capacitance(S, N, l, t, d=0, gap=None, V_pos=3, V_neg=0):
    """
    Compute the capacitance C = Q / V for an interdigitated capacitor.

    Returns
    -------
    C : float - capacitance in grid units (epsilon_0 * pixels)
    """
    if gap is None:
        delta = S // N
    else:
        delta = t + gap

    potential = interdigitated_potential(S, N, l, t, d=d, gap=gap, V_pos=V_pos, V_neg=V_neg)
    Ex, Ey = potential.gradient()

    pos_mask = np.zeros((S, S), dtype=bool)
    pos_mask[0:S, 0:3] = True
    for i in range(N):
        pos_mask[i * delta: i * delta + t, 0:l] = True

    outline_pos, nx_pos, ny_pos = potential.boundary_outline(pos_mask)
    # E = -grad(V), sigma = epsilon_0 * E·n = -grad(V)·n
    sigma_pos = -(Ex[outline_pos] * nx_pos[outline_pos] + Ey[outline_pos] * ny_pos[outline_pos])
    Q_pos = np.sum(sigma_pos)

    V = V_pos - V_neg
    return Q_pos / V


class TestUtilities(unittest.TestCase):
    def test_interdigitated_potential_solves(self):
        """interdigitated_potential returns a solved ScalarField."""
        S, N, t = 60, 3, 2
        l = int(0.95 * S)
        potential = interdigitated_potential(S, N, l, t)
        self.assertEqual(potential.shape, (S, S))
        self.assertFalse(np.all(potential.values == 0))

    def test_boundary_conditions_applied(self):
        """Boundary values match V_pos and V_neg on the plates."""
        S, N, t = 60, 3, 2
        l = int(0.95 * S)
        V_pos, V_neg = 5, 0
        potential = interdigitated_potential(S, N, l, t, V_pos=V_pos, V_neg=V_neg)
        np.testing.assert_allclose(potential.values[:, 0], V_pos, atol=1e-3)
        np.testing.assert_allclose(potential.values[:, -1], V_neg, atol=1e-3)

    def test_boundary_mask_not_empty(self):
        """boundary_mask covers all boundary condition pixels."""
        S, N, t = 60, 3, 2
        l = int(0.95 * S)
        potential = interdigitated_potential(S, N, l, t)
        mask = potential.boundary_mask
        self.assertTrue(np.any(mask))

    def test_boundary_outline_surrounds_mask(self):
        """Outline pixels are adjacent to but outside the mask."""
        S, N, t = 60, 3, 2
        l = int(0.95 * S)
        potential = interdigitated_potential(S, N, l, t)
        mask = potential.boundary_mask
        outline, nx, ny = potential.boundary_outline()
        # Outline should not overlap the mask
        self.assertFalse(np.any(outline & mask))
        # Outline should have nonzero normals
        norm = np.sqrt(nx[outline]**2 + ny[outline]**2)
        self.assertTrue(np.all(norm > 0))

    def test_charge_conservation(self):
        """Total charge (Q_pos + Q_neg) should be approximately zero."""
        S, N, t = 100, 5, 2
        l = int(0.95 * S)
        delta = S // N
        offset = delta // 2

        potential = interdigitated_potential(S, N, l, t)
        Ex, Ey = potential.gradient()

        pos_mask = np.zeros((S, S), dtype=bool)
        pos_mask[0:S, 0:3] = True
        for i in range(N):
            pos_mask[i * delta: i * delta + t, 0:l] = True

        neg_mask = np.zeros((S, S), dtype=bool)
        neg_mask[0:S, S - 3:S] = True
        for i in range(N):
            neg_mask[offset + i * delta: offset + i * delta + t, S - l:S] = True

        outline_pos, nx_pos, ny_pos = potential.boundary_outline(pos_mask)
        Q_pos = np.sum(-(Ex[outline_pos] * nx_pos[outline_pos] + Ey[outline_pos] * ny_pos[outline_pos]))

        outline_neg, nx_neg, ny_neg = potential.boundary_outline(neg_mask)
        Q_neg = np.sum(-(Ex[outline_neg] * nx_neg[outline_neg] + Ey[outline_neg] * ny_neg[outline_neg]))

        relative_error = abs(Q_pos + Q_neg) / abs(Q_pos)
        self.assertLess(relative_error, 0.05)

    def test_capacitance_positive(self):
        """Capacitance should be positive."""
        S, N, t = 60, 3, 2
        l = int(0.95 * S)
        C = interdigitated_capacitance(S, N, l, t)
        self.assertGreater(C, 0)

    def test_capacitance_scales_with_voltage(self):
        """C = Q/V should not depend on the applied voltage."""
        S, N, t = 60, 3, 2
        l = int(0.95 * S)
        C1 = interdigitated_capacitance(S, N, l, t, V_pos=3, V_neg=0)
        C2 = interdigitated_capacitance(S, N, l, t, V_pos=6, V_neg=0)
        np.testing.assert_allclose(C1, C2, rtol=0.02)


class TestAccelerometer(unittest.TestCase):
    def test01_capacitance_per_length_vs_N(self):
        """C/l as a function of N, with fixed finger spacing (S scales with N)."""
        t = 2
        delta = 20

        Ns = []
        Cs_per_l = []
        print(f"\nN\tS\tl\tC\tC/l")
        for N in range(3, 20):
            S = N * delta
            l = int(0.95 * S)

            C = interdigitated_capacitance(S, N, l, t)
            C_per_l = C / l

            Ns.append(N)
            Cs_per_l.append(C_per_l)
            print(f"{N}\t{S}\t{l}\t{C:.4f}\t{C_per_l:.4f}")

        plt.plot(Ns, Cs_per_l, 'o-')
        plt.xlabel("Number of fingers N")
        plt.ylabel("C / l (capacitance per unit finger length)")
        plt.title("C/l vs N (fixed spacing)")
        plt.grid(True)
        plt.show()


    def test02_capacitance_per_length_vs_spacing(self):
        """C/l as a function of finger spacing delta, with N=10 fixed."""
        N = 10
        t = 2

        deltas = []
        Cs_per_l = []
        print(f"\ndelta\tS\tl\tC\tC/l")
        for delta in range(t + 3, 40):
            S = N * delta
            l = int(0.95 * S)

            C = interdigitated_capacitance(S, N, l, t)
            C_per_l = C / l

            deltas.append(delta)
            Cs_per_l.append(C_per_l)
            print(f"{delta}\t{S}\t{l}\t{C:.4f}\t{C_per_l:.4f}")

        plt.plot(deltas, Cs_per_l, 'o-')
        plt.xlabel("Finger spacing delta (pixels)")
        plt.ylabel("C / l (capacitance per unit finger length)")
        plt.title("C/l vs spacing (N=10)")
        plt.grid(True)
        plt.show()


    def test03_capacitance_per_length_vs_thickness(self):
        """C/l as a function of finger thickness t, with fixed gap."""
        N = 10
        gap = 10
        S = 400
        l = int(0.95 * S)

        ts = []
        Cs_per_l = []
        print(f"\nt\tgap\tC\tC/l")
        t_max = gap // 2
        for t in range(1, t_max + 1):
            C = interdigitated_capacitance(S, N, l, t, gap=gap)
            C_per_l = C / l

            ts.append(t)
            Cs_per_l.append(C_per_l)
            print(f"{t}\t{gap}\t{C:.4f}\t{C_per_l:.4f}")

        plt.plot(ts, Cs_per_l, 'o-')
        plt.xlabel("Finger thickness t (pixels)")
        plt.ylabel("C / l (capacitance per unit finger length)")
        plt.title(f"C/l vs thickness (N={N}, gap={gap}, S={S})")
        plt.grid(True)
        plt.show()


def interdigitated_potential_3d(Sx, Sy, Sz, N, l, t, h, d=0, gap=None, V_pos=3, V_neg=0):
    """
    Build a 3D ScalarField for an interdigitated capacitor and solve Laplace.
    The fingers are thin plates that float in the volume.

    Parameters
    ----------
    Sx, Sy, Sz : int - grid size along x (between plates), y (along fingers), z (height)
    N : int - number of finger pairs
    l : int - finger length in pixels (along y)
    t : int - finger thickness in pixels (along x, spacing direction)
    h : int - finger height in pixels (along z)
    d : int - lateral offset of negative fingers
    gap : int or None - gap between fingers along x
    V_pos, V_neg : float - voltages
    """
    if gap is None:
        delta = Sx // N
    else:
        delta = t + gap

    offset = delta // 2 + d
    z0 = (Sz - h) // 2  # center fingers vertically

    potential = ScalarField(shape=(Sx, Sy, Sz))

    # Side plates (full yz planes)
    potential.add_boundary_condition((slice(0, 3), slice(None), slice(None)), V_pos)
    potential.add_boundary_condition((slice(Sx - 3, Sx), slice(None), slice(None)), V_neg)

    # Positive fingers: thin plates from y=0
    for i in range(N):
        finger = (slice(i * delta, i * delta + t), slice(0, l), slice(z0, z0 + h))
        potential.add_boundary_condition(finger, V_pos)

    # Negative fingers: thin plates from y=Sy
    for i in range(N):
        finger = (slice(offset + i * delta, offset + i * delta + t),
                  slice(Sy - l, Sy), slice(z0, z0 + h))
        potential.add_boundary_condition(finger, V_neg)

    potential.apply_conditions()
    potential.solve_laplace_by_relaxation()
    return potential


def interdigitated_capacitance_3d(Sx, Sy, Sz, N, l, t, h, d=0, gap=None, V_pos=3, V_neg=0):
    """Compute capacitance C = Q / V for the 3D interdigitated capacitor."""
    if gap is None:
        delta = Sx // N
    else:
        delta = t + gap

    z0 = (Sz - h) // 2

    potential = interdigitated_potential_3d(Sx, Sy, Sz, N, l, t, h,
                                           d=d, gap=gap, V_pos=V_pos, V_neg=V_neg)
    Ex, Ey, Ez = potential.gradient()

    pos_mask = np.zeros((Sx, Sy, Sz), dtype=bool)
    pos_mask[0:3, :, :] = True
    for i in range(N):
        pos_mask[i * delta: i * delta + t, 0:l, z0:z0 + h] = True

    outline, nx, ny, nz = potential.boundary_outline(pos_mask)
    # E = -grad(V), sigma = epsilon_0 * E·n = -grad(V)·n
    sigma = -(Ex[outline] * nx[outline]
              + Ey[outline] * ny[outline]
              + Ez[outline] * nz[outline])
    Q_pos = np.sum(sigma)

    V = V_pos - V_neg
    return Q_pos / V


class TestAccelerometer3D(unittest.TestCase):
    def test01_3d_solve_and_visualize(self):
        """Solve a small 3D interdigitated capacitor and show slices."""
        Sx, Sy, Sz = 60, 60, 30
        N, t, h = 3, 2, 10
        l = int(0.95 * Sy)

        potential = interdigitated_potential_3d(Sx, Sy, Sz, N, l, t, h)
        mask = potential.boundary_mask

        mid_z = Sz // 2
        mid_y = Sy // 2
        mid_x = Sx // 2

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        # Rangee du haut: masques
        axes[0, 0].imshow(mask[:, :, mid_z].T, origin='lower', cmap='gray')
        axes[0, 0].set_title(f"Masque z={mid_z} (dessus)")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")

        axes[0, 1].imshow(mask[:, mid_y, :].T, origin='lower', cmap='gray')
        axes[0, 1].set_title(f"Masque y={mid_y} (cote)")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("z")

        axes[0, 2].imshow(mask[mid_x, :, :].T, origin='lower', cmap='gray')
        axes[0, 2].set_title(f"Masque x={mid_x} (face)")
        axes[0, 2].set_xlabel("y")
        axes[0, 2].set_ylabel("z")

        # Rangee du bas: potentiel
        axes[1, 0].imshow(potential.values[:, :, mid_z].T, origin='lower', cmap='RdBu_r')
        axes[1, 0].set_title(f"Potentiel z={mid_z} (dessus)")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("y")

        axes[1, 1].imshow(potential.values[:, mid_y, :].T, origin='lower', cmap='RdBu_r')
        axes[1, 1].set_title(f"Potentiel y={mid_y} (cote)")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("z")

        axes[1, 2].imshow(potential.values[mid_x, :, :].T, origin='lower', cmap='RdBu_r')
        axes[1, 2].set_title(f"Potentiel x={mid_x} (face)")
        axes[1, 2].set_xlabel("y")
        axes[1, 2].set_ylabel("z")

        plt.tight_layout()
        plt.show()

    def test02_3d_capacitance(self):
        """Compute the capacitance of the 3D accelerometer."""
        Sx, Sy, Sz = 60, 60, 30
        N, t, h = 3, 2, 10
        l = int(0.95 * Sy)

        C = interdigitated_capacitance_3d(Sx, Sy, Sz, N, l, t, h)
        print(f"\n3D Capacitance: C = {C:.4f}")
        self.assertGreater(C, 0)

    def test03_3d_capacitance_vs_height(self):
        """C vs finger height h: thinner fingers in z should have less capacitance."""
        Sx, Sy, Sz = 60, 60, 30
        N, t = 3, 2
        l = int(0.95 * Sy)

        hs = []
        Cs = []
        print(f"\nh\tC")
        for h in range(2, Sz - 4, 2):
            C = interdigitated_capacitance_3d(Sx, Sy, Sz, N, l, t, h)
            hs.append(h)
            Cs.append(C)
            print(f"{h}\t{C:.4f}")

        plt.plot(hs, Cs, 'o-')
        plt.xlabel("Finger height h (pixels)")
        plt.ylabel("Capacitance C")
        plt.title("3D: C vs finger height")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    unittest.main()
