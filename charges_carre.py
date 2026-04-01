"""
Simulation 2D : charges positives unitaires confinées dans un carré.
Chaque charge crée un champ électrique E = k*q/r² qui repousse les autres.
Les charges ne peuvent pas sortir du carré.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import pdist, squareform

# Paramètres
N_CHARGES = 50
COTE = 2.0              # carré de -1 à +1
DEMI_COTE = COTE / 2
K_COULOMB = 0.5
DT = 0.005
AMORTISSEMENT = 0.85
Q = 1.0
R_MIN = 0.02
BRUIT = 5.0

# Placement aléatoire dans le carré
positions = np.random.uniform(-DEMI_COTE * 0.8, DEMI_COTE * 0.8, (N_CHARGES, 2))
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

def confiner_dans_carre(pos, vel, demi_cote):
    """Empêche les charges de sortir du carré (rebond élastique)."""
    for axis in range(2):
        # Dépassement à droite/haut
        trop_haut = pos[:, axis] > demi_cote
        pos[trop_haut, axis] = demi_cote
        vel[trop_haut, axis] = np.minimum(vel[trop_haut, axis], 0)

        # Dépassement à gauche/bas
        trop_bas = pos[:, axis] < -demi_cote
        pos[trop_bas, axis] = -demi_cote
        vel[trop_bas, axis] = np.maximum(vel[trop_bas, axis], 0)

# --- Visualisation ---
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.set_aspect('equal')
ax.set_title(f'{N_CHARGES} charges positives confinées dans un carré', fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Dessiner le carré
carre = plt.Rectangle((-DEMI_COTE, -DEMI_COTE), COTE, COTE,
                       fill=False, color='black', linewidth=2)
ax.add_patch(carre)

# Points pour les charges
scatter = ax.scatter(positions[:, 0], positions[:, 1],
                     s=40, c='red', edgecolors='darkred', zorder=5, linewidths=1)

# Marqueurs "+"
plus_texts = []
for i in range(N_CHARGES):
    t = ax.text(positions[i, 0], positions[i, 1], '+',
                ha='center', va='center', fontsize=7, fontweight='bold', color='white', zorder=6)
    plus_texts.append(t)

info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    va='top', fontsize=10, family='monospace')

step_count = [0]

def update(frame):
    global positions, vitesses

    for _ in range(3):
        forces = calculer_forces(positions)
        f_norm = np.linalg.norm(forces, axis=1, keepdims=True)
        f_max = 50.0
        forces = np.where(f_norm > f_max, forces * f_max / f_norm, forces)

        bruit = np.random.randn(N_CHARGES, 2) * BRUIT
        vitesses += (forces + bruit) * DT
        vitesses *= AMORTISSEMENT
        positions += vitesses * DT
        confiner_dans_carre(positions, vitesses, DEMI_COTE)

    step_count[0] += 3

    scatter.set_offsets(positions)
    for i, t in enumerate(plus_texts):
        t.set_position((positions[i, 0], positions[i, 1]))

    Ec = 0.5 * np.sum(vitesses**2)
    info_text.set_text(f'Pas: {step_count[0]}  Ec: {Ec:.2f}')

    return scatter, *plus_texts, info_text

ani = FuncAnimation(fig, update, frames=range(2000), interval=30, blit=False)
plt.tight_layout()
plt.show()
