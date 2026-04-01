"""
Simulation 2D : charges positives unitaires confinées dans un cercle.
Chaque charge crée un champ électrique E = k*q/r² qui repousse les autres.
Les charges ne peuvent pas sortir du cercle.
Optimisé pour N=1000 avec scipy.spatial.distance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import pdist, squareform

# Paramètres
N_CHARGES = 10000
RAYON_CERCLE = 1.0
K_COULOMB = 0.5
DT = 0.005
AMORTISSEMENT = 0.85
Q = 1.0
R_MIN = 0.02
BRUIT = 5.0             # amplitude du bruit thermique (agitation aléatoire)

# Placement aléatoire dans le cercle
def positions_aleatoires(n, rayon):
    pos = np.empty((0, 2))
    while len(pos) < n:
        batch = np.random.uniform(-1, 1, (n * 2, 2))
        inside = np.linalg.norm(batch, axis=1) <= 1.0
        pos = np.vstack([pos, batch[inside]])
    return pos[:n] * rayon * 0.8

positions = positions_aleatoires(N_CHARGES, RAYON_CERCLE)
vitesses = np.zeros((N_CHARGES, 2))

def calculer_forces(pos):
    """Calcule les forces coulombiennes avec scipy."""
    dist = squareform(pdist(pos))
    dist = np.maximum(dist, R_MIN)
    np.fill_diagonal(dist, 1.0)

    inv_r3 = K_COULOMB * Q * Q / (dist ** 3)
    np.fill_diagonal(inv_r3, 0.0)

    sum_inv_r3 = inv_r3.sum(axis=1)
    forces = pos * sum_inv_r3[:, np.newaxis] - inv_r3 @ pos
    return forces

def confiner_dans_cercle(pos, vel, rayon):
    """Empêche les charges de sortir du cercle (vectorisé)."""
    r = np.linalg.norm(pos, axis=1)
    dehors = r > rayon
    if not np.any(dehors):
        return
    pos[dehors] = pos[dehors] / r[dehors, np.newaxis] * rayon
    direction = pos[dehors] / rayon
    v_rad = np.sum(vel[dehors] * direction, axis=1)
    sortant = v_rad > 0
    if np.any(sortant):
        idx = np.where(dehors)[0][sortant]
        d = direction[sortant]
        vel[idx] -= np.sum(vel[idx] * d, axis=1)[:, np.newaxis] * d

# --- Visualisation ---
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')
ax.set_title(f'{N_CHARGES} charges positives confinées dans un cercle', fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Dessiner le cercle
cercle = plt.Circle((0, 0), RAYON_CERCLE, fill=False, color='black', linewidth=2)
ax.add_patch(cercle)

# Points pour les charges
scatter = ax.scatter(positions[:, 0], positions[:, 1],
                     s=3, c='red', alpha=0.7, zorder=5)

info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    va='top', fontsize=10, family='monospace')

step_count = [0]

def update(frame):
    global positions, vitesses

    for _ in range(1):
        forces = calculer_forces(positions)
        # Limiter la force max pour stabilité
        f_norm = np.linalg.norm(forces, axis=1, keepdims=True)
        f_max = 50.0
        forces = np.where(f_norm > f_max, forces * f_max / f_norm, forces)

        # Ajouter un bruit thermique aléatoire (évite la cristallisation en anneaux)
        bruit = np.random.randn(N_CHARGES, 2) * BRUIT
        vitesses += (forces + bruit) * DT
        vitesses *= AMORTISSEMENT
        positions += vitesses * DT
        confiner_dans_cercle(positions, vitesses, RAYON_CERCLE)

    step_count[0] += 1

    scatter.set_offsets(positions)

    Ec = 0.5 * np.sum(vitesses**2)
    r_all = np.linalg.norm(positions, axis=1)
    info_text.set_text(f'Pas: {step_count[0]}  Ec: {Ec:.2f}  r_moy: {r_all.mean():.3f}  r_max: {r_all.max():.3f}')

    return scatter, info_text

ani = FuncAnimation(fig, update, frames=range(2000), interval=100, blit=False)
plt.tight_layout()
plt.show()
