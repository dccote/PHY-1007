"""
Simulation 2D : charges positives confinées dans une forme irrégulière asymétrique.
La forme est un polygone quelconque avec des pointes et des parties arrondies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import pdist, squareform
from matplotlib.path import Path

# Paramètres
N_CHARGES = 3000
K_COULOMB = 0.5
DT = 0.005
AMORTISSEMENT = 0.85
Q = 1.0
R_MIN = 0.02
BRUIT = 5.0

# Forme irrégulière : un polygone asymétrique avec une pointe aiguë
SOMMETS = np.array([
    [-0.3, 1.2],       # pointe aiguë en haut-gauche
    [-1.1, -0.5],      # bas-gauche
    [0.5, -0.9],       # bas-droite
    [1.3, 0.3],        # droite large
])

# Fermer le polygone
poly_path = Path(np.vstack([SOMMETS, SOMMETS[0]]))

# Arêtes avec normales intérieures
n_edges = len(SOMMETS)
EDGES = []
centroid = SOMMETS.mean(axis=0)
for i in range(n_edges):
    p1 = SOMMETS[i]
    p2 = SOMMETS[(i + 1) % n_edges]
    edge = p2 - p1
    normal = np.array([-edge[1], edge[0]])
    normal = normal / np.linalg.norm(normal)
    # S'assurer que la normale pointe vers l'intérieur (vers le centroïde)
    if np.dot(normal, centroid - p1) < 0:
        normal = -normal
    EDGES.append((p1, p2, normal))

# Placement aléatoire dans la forme
def positions_aleatoires(n):
    pos = []
    xmin, xmax = SOMMETS[:, 0].min(), SOMMETS[:, 0].max()
    ymin, ymax = SOMMETS[:, 1].min(), SOMMETS[:, 1].max()
    while len(pos) < n:
        p = np.array([np.random.uniform(xmin, xmax),
                       np.random.uniform(ymin, ymax)])
        if poly_path.contains_point(p):
            pos.append(p)
    return np.array(pos)

positions = positions_aleatoires(N_CHARGES)
vitesses = np.zeros((N_CHARGES, 2))

def calculer_forces(pos):
    dist = squareform(pdist(pos))
    dist = np.maximum(dist, R_MIN)
    np.fill_diagonal(dist, 1.0)
    inv_r3 = K_COULOMB * Q * Q / (dist ** 3)
    np.fill_diagonal(inv_r3, 0.0)
    sum_inv_r3 = inv_r3.sum(axis=1)
    forces = pos * sum_inv_r3[:, np.newaxis] - inv_r3 @ pos
    return forces

def confiner_dans_forme(pos, vel):
    """Empêche les charges de sortir du polygone."""
    for i in range(len(pos)):
        for p1, p2, normal in EDGES:
            d = np.dot(pos[i] - p1, normal)
            if d < 0:
                pos[i] -= d * normal
                v_n = np.dot(vel[i], normal)
                if v_n < 0:
                    vel[i] -= v_n * normal

# --- Visualisation ---
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_xlim(-1.6, 1.8)
ax.set_ylim(-1.5, 1.7)
ax.set_aspect('equal')
ax.set_title(f'{N_CHARGES} charges dans une forme irrégulière', fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Dessiner la forme
forme = plt.Polygon(SOMMETS, fill=False, color='black', linewidth=2)
ax.add_patch(forme)

scatter = ax.scatter(positions[:, 0], positions[:, 1],
                     s=3, c='red', alpha=0.7, zorder=5)

plus_texts = []

info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                    va='top', fontsize=10, family='monospace')

step_count = [0]

def update(frame):
    global positions, vitesses

    for _ in range(1):
        forces = calculer_forces(positions)
        f_norm = np.linalg.norm(forces, axis=1, keepdims=True)
        f_max = 50.0
        forces = np.where(f_norm > f_max, forces * f_max / f_norm, forces)

        bruit = np.random.randn(N_CHARGES, 2) * BRUIT
        vitesses += (forces + bruit) * DT
        vitesses *= AMORTISSEMENT
        positions += vitesses * DT
        confiner_dans_forme(positions, vitesses)

    step_count[0] += 1

    scatter.set_offsets(positions)
    for i, t in enumerate(plus_texts):
        t.set_position((positions[i, 0], positions[i, 1]))

    Ec = 0.5 * np.sum(vitesses**2)
    info_text.set_text(f'Pas: {step_count[0]}  Ec: {Ec:.2f}')

    return scatter, *plus_texts, info_text

ani = FuncAnimation(fig, update, frames=range(2000), interval=30, blit=False)
plt.tight_layout()
plt.show()
