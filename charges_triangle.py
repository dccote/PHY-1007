"""
Simulation 2D : charges positives unitaires confinées dans un triangle équilatéral.
Chaque charge crée un champ électrique E = k*q/r² qui repousse les autres.
Les charges ne peuvent pas sortir du triangle.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import pdist, squareform

# Paramètres
N_CHARGES = 50
K_COULOMB = 0.5
DT = 0.005
AMORTISSEMENT = 0.85
Q = 1.0
R_MIN = 0.02
BRUIT = 5.0

# Triangle équilatéral centré à l'origine
# Sommets à distance 1.2 du centre
R_TRI = 1.2
SOMMETS = np.array([
    [R_TRI * np.cos(np.pi/2), R_TRI * np.sin(np.pi/2)],
    [R_TRI * np.cos(np.pi/2 + 2*np.pi/3), R_TRI * np.sin(np.pi/2 + 2*np.pi/3)],
    [R_TRI * np.cos(np.pi/2 + 4*np.pi/3), R_TRI * np.sin(np.pi/2 + 4*np.pi/3)],
])

# Arêtes du triangle : chaque arête définie par un point et une normale intérieure
EDGES = []
for i in range(3):
    p1 = SOMMETS[i]
    p2 = SOMMETS[(i + 1) % 3]
    edge = p2 - p1
    # Normale pointant vers l'intérieur
    normal = np.array([-edge[1], edge[0]])
    # S'assurer qu'elle pointe vers le centre (0,0)
    centre_dir = -0.5 * (p1 + p2)
    if np.dot(normal, centre_dir) < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    EDGES.append((p1, p2, normal))

def point_dans_triangle(p):
    """Vérifie si un point est dans le triangle via les normales."""
    for p1, p2, normal in EDGES:
        if np.dot(p - p1, normal) < 0:
            return False
    return True

# Placement aléatoire dans le triangle
def positions_aleatoires(n):
    pos = []
    # Bounding box du triangle
    xmin, xmax = SOMMETS[:, 0].min(), SOMMETS[:, 0].max()
    ymin, ymax = SOMMETS[:, 1].min(), SOMMETS[:, 1].max()
    while len(pos) < n:
        p = np.array([np.random.uniform(xmin, xmax),
                       np.random.uniform(ymin, ymax)])
        if point_dans_triangle(p * 1.1):  # un peu de marge
            pos.append(p * 0.8)
    return np.array(pos)

positions = positions_aleatoires(N_CHARGES)
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

def confiner_dans_triangle(pos, vel):
    """Empêche les charges de sortir du triangle."""
    for i in range(len(pos)):
        for p1, p2, normal in EDGES:
            # Distance signée au mur (positive = intérieur)
            d = np.dot(pos[i] - p1, normal)
            if d < 0:
                # Repousser sur le mur
                pos[i] -= d * normal
                # Annuler la composante de vitesse vers l'extérieur
                v_n = np.dot(vel[i], normal)
                if v_n < 0:
                    vel[i] -= v_n * normal

# --- Visualisation ---
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.2, 1.6)
ax.set_aspect('equal')
ax.set_title(f'{N_CHARGES} charges positives confinées dans un triangle', fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Dessiner le triangle
triangle = plt.Polygon(SOMMETS, fill=False, color='black', linewidth=2)
ax.add_patch(triangle)

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
        confiner_dans_triangle(positions, vitesses)

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
