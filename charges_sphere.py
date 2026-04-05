"""
Simulation 3D : charges positives unitaires confinées dans une sphère.
Chaque charge crée un champ électrique E = k*q/r² qui repousse les autres.
Les charges ne peuvent pas sortir de la sphère.
Optimisé pour N=1000 avec scipy.spatial.distance.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import pdist, squareform

# Paramètres
N_CHARGES = 1000
RAYON_SPHERE = 1.0
K_COULOMB = 0.5
DT = 0.001
AMORTISSEMENT = 0.85
Q = 1.0
R_MIN = 0.03

# Placement aléatoire dans la sphère
def positions_aleatoires(n, rayon):
    # Méthode vectorisée : rejection sampling par lots
    pos = np.empty((0, 3))
    while len(pos) < n:
        batch = np.random.uniform(-1, 1, (n * 2, 3))
        inside = np.linalg.norm(batch, axis=1) <= 1.0
        pos = np.vstack([pos, batch[inside]])
    return pos[:n] * rayon * 0.8

positions = positions_aleatoires(N_CHARGES, RAYON_SPHERE)
vitesses = np.zeros((N_CHARGES, 3))

def calculer_forces(pos):
    """Calcule les forces coulombiennes avec scipy (plus efficace en mémoire)."""
    n = len(pos)
    # Matrice de distances condensée puis carrée
    dist = squareform(pdist(pos))
    dist = np.maximum(dist, R_MIN)
    np.fill_diagonal(dist, 1.0)  # éviter div/0

    # F = k*q²/r³ * (ri - rj) => on calcule k*q²/r³ puis multiplie par diff
    inv_r3 = K_COULOMB * Q * Q / (dist ** 3)
    np.fill_diagonal(inv_r3, 0.0)

    # Force sur chaque charge
    # forces[i] = sum_j inv_r3[i,j] * (pos[i] - pos[j])
    # = pos[i] * sum_j(inv_r3[i,j]) - sum_j(inv_r3[i,j] * pos[j])
    sum_inv_r3 = inv_r3.sum(axis=1)  # (n,)
    forces = pos * sum_inv_r3[:, np.newaxis] - inv_r3 @ pos  # (n, 3)
    return forces

def confiner_dans_sphere(pos, vel, rayon):
    """Empêche les charges de sortir de la sphère (vectorisé)."""
    r = np.linalg.norm(pos, axis=1)
    dehors = r > rayon
    if not np.any(dehors):
        return
    # Ramener sur le bord
    pos[dehors] = pos[dehors] / r[dehors, np.newaxis] * rayon
    # Annuler composante radiale sortante
    direction = pos[dehors] / rayon
    v_rad = np.sum(vel[dehors] * direction, axis=1)
    sortant = v_rad > 0
    if np.any(sortant):
        idx = np.where(dehors)[0][sortant]
        d = direction[sortant]
        vel[idx] -= np.sum(vel[idx] * d, axis=1)[:, np.newaxis] * d

# --- Visualisation 3D ---
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_zlim(-1.3, 1.3)
ax.set_title(f'{N_CHARGES} charges positives confinées dans une sphère', fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Sphère wireframe
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
xs = RAYON_SPHERE * np.outer(np.cos(u), np.sin(v))
ys = RAYON_SPHERE * np.outer(np.sin(u), np.sin(v))
zs = RAYON_SPHERE * np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(xs, ys, zs, color='steelblue', alpha=0.08, linewidth=0.5)

# Points pour les charges
scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                     s=5, c='red', alpha=0.7, depthshade=True, zorder=5)

info_text = fig.text(0.02, 0.98, '', va='top', fontsize=10, family='monospace')

step_count = [0]

def update(frame):
    global positions, vitesses

    # 2 sous-pas par frame (compromis vitesse/précision)
    for _ in range(2):
        forces = calculer_forces(positions)
        # Limiter la force max pour stabilité
        f_norm = np.linalg.norm(forces, axis=1, keepdims=True)
        f_max = 50.0
        forces = np.where(f_norm > f_max, forces * f_max / f_norm, forces)

        vitesses += forces * DT
        vitesses *= AMORTISSEMENT
        positions += vitesses * DT
        confiner_dans_sphere(positions, vitesses, RAYON_SPHERE)

    step_count[0] += 2

    # Mettre à jour les positions
    scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

    # Rotation lente
    ax.view_init(elev=25, azim=frame * 0.5)

    # Stats
    Ec = 0.5 * np.sum(vitesses**2)
    r_all = np.linalg.norm(positions, axis=1)
    info_text.set_text(f'Pas: {step_count[0]}  Ec: {Ec:.2f}  r_moy: {r_all.mean():.3f}  r_max: {r_all.max():.3f}')

    return scatter, info_text

ani = FuncAnimation(fig, update, frames=range(2000), interval=50, blit=False)
plt.tight_layout()
plt.show()
