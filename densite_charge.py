"""
Animation 2D : densité de charge σ sur le contour d'un conducteur.
On part d'une distribution uniforme de σ, puis on la fait évoluer
jusqu'à ce que le potentiel soit constant sur le contour (équilibre).
La couleur et l'épaisseur des segments montrent σ en temps réel.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

# --- Choix de la forme ---
FORME = 'pointe'  # 'cercle', 'carre', 'triangle', 'irregulier', 'pointe'
N_SEGMENTS = 150
DT = 0.05
AMORTISSEMENT = 0.7

def forme_cercle(n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta)])

def forme_carre(n):
    n4 = n // 4
    pts = []
    for i in range(n4): pts.append([1 - 2*i/n4, 1])
    for i in range(n4): pts.append([-1, 1 - 2*i/n4])
    for i in range(n4): pts.append([-1 + 2*i/n4, -1])
    for i in range(n - 3*n4): pts.append([1, -1 + 2*i/(n - 3*n4)])
    return np.array(pts)

def forme_triangle(n):
    n3 = n // 3
    sommets = [[0, 1.2], [-1.04, -0.6], [1.04, -0.6]]
    pts = []
    for s in range(3):
        p1, p2 = np.array(sommets[s]), np.array(sommets[(s+1) % 3])
        nn = n3 if s < 2 else n - 2*n3
        for i in range(nn):
            pts.append(p1 + i/nn * (p2 - p1))
    return np.array(pts)

def forme_irreguliere(n):
    sommets = np.array([[-0.3, 1.2], [-1.1, -0.5], [0.5, -0.9], [1.3, 0.3]])
    lengths = [np.linalg.norm(sommets[(i+1)%4] - sommets[i]) for i in range(4)]
    total = sum(lengths)
    pts = []
    for s in range(4):
        p1, p2 = sommets[s], sommets[(s+1) % 4]
        ns = max(2, int(n * lengths[s] / total))
        for i in range(ns):
            pts.append(p1 + i/ns * (p2 - p1))
    return np.array(pts[:n])

def forme_pointe(n):
    n_base = int(n * 0.65)
    n_pointe = n - n_base
    pts = []
    for i in range(n_base):
        theta = np.pi + i/n_base * np.pi
        pts.append([1.0 * np.cos(theta), 0.5 * np.sin(theta)])
    half = n_pointe // 2
    for i in range(half):
        t = i / half
        pts.append([1.0*(1-t), t*2.0])
    for i in range(n_pointe - half):
        t = i / (n_pointe - half)
        pts.append([-1.0*t, 2.0*(1-t)])
    return np.array(pts)

# Sélection de la forme
formes_dict = {
    'cercle': forme_cercle,
    'carre': forme_carre,
    'triangle': forme_triangle,
    'irregulier': forme_irreguliere,
    'pointe': forme_pointe,
}
points = formes_dict[FORME](N_SEGMENTS)
n = len(points)

# Centres et longueurs des segments
centres = np.zeros((n, 2))
dl = np.zeros(n)
for i in range(n):
    centres[i] = 0.5 * (points[i] + points[(i+1) % n])
    dl[i] = np.linalg.norm(points[(i+1) % n] - points[i])

# Matrice d'influence (potentiel au segment i dû au segment j)
A = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            A[i, j] = -dl[j] * (np.log(dl[j]/2) - 1) / (2 * np.pi)
        else:
            r = np.linalg.norm(centres[i] - centres[j])
            A[i, j] = -dl[j] * np.log(r) / (2 * np.pi)

# σ initial : uniforme
sigma = np.ones(n) / (np.sum(dl))  # charge totale = 1
Q_total = np.sum(sigma * dl)

# --- Visualisation ---
fig, (ax_forme, ax_graph) = plt.subplots(1, 2, figsize=(14, 6))

# Panneau gauche : la forme avec couleur σ
segments_plot = [[points[i], points[(i+1) % n]] for i in range(n)]
norm_color = Normalize(vmin=0, vmax=1)
lc = LineCollection(segments_plot, cmap='hot_r', norm=norm_color, linewidths=4)
lc.set_array(np.abs(sigma))
ax_forme.add_collection(lc)

margin = 0.4
all_pts = np.array(points)
ax_forme.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
ax_forme.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
ax_forme.set_aspect('equal')
ax_forme.set_title(f'Conducteur ({FORME}) — densité σ', fontsize=13)
ax_forme.set_xlabel('x')
ax_forme.set_ylabel('y')
cb = plt.colorbar(lc, ax=ax_forme, fraction=0.046, pad=0.04)
cb.set_label('σ')

# Panneau droit : σ vs position sur le contour
arc = np.cumsum(dl) - dl[0]
arc_norm = arc / arc[-1]
line_sigma, = ax_graph.plot(arc_norm, sigma, 'r-', linewidth=2)
fill_sigma = ax_graph.fill_between(arc_norm, sigma, alpha=0.3, color='red')
ax_graph.set_xlabel('Position sur le contour (normalisée)')
ax_graph.set_ylabel('σ (densité de charge)')
ax_graph.set_title('σ le long du contour', fontsize=13)
ax_graph.set_xlim(0, 1)

info_text = fig.text(0.5, 0.02, '', ha='center', fontsize=11, family='monospace')

step_count = [0]

def update(frame):
    global sigma, fill_sigma

    # Calculer le potentiel actuel sur chaque segment
    V = A @ sigma  # potentiel en chaque point du contour

    # Le gradient de V nous dit comment ajuster σ
    # On veut V = constante partout => on pousse σ là où V est bas
    V_mean = np.mean(V)
    dV = V - V_mean

    # Ajuster σ : diminuer là où V est trop haut, augmenter là où V est trop bas
    sigma -= DT * dV * AMORTISSEMENT

    # Garder σ >= 0 (charges positives seulement)
    sigma = np.maximum(sigma, 0.0)

    # Renormaliser pour garder la charge totale constante
    Q = np.sum(sigma * dl)
    if Q > 0:
        sigma *= Q_total / Q

    step_count[0] += 1

    # Mettre à jour la couleur sur la forme
    sigma_abs = np.abs(sigma)
    s_max = max(sigma_abs.max(), 1e-10)
    norm_color.vmin = 0
    norm_color.vmax = s_max
    lc.set_array(sigma_abs)
    lc.set_linewidths(2 + 6 * sigma_abs / s_max)

    # Mettre à jour le graphique σ
    line_sigma.set_ydata(sigma)
    ax_graph.set_ylim(0, s_max * 1.2)

    # Mettre à jour le fill
    fill_sigma.remove()
    fill_sigma = ax_graph.fill_between(arc_norm, sigma, alpha=0.3, color='red')

    # Écart-type du potentiel (mesure de convergence)
    V_std = np.std(V)
    info_text.set_text(f'Pas: {step_count[0]}  |  écart-type(V): {V_std:.6f}  |  σ_max: {sigma.max():.4f}  σ_min: {sigma.min():.4f}')

    return lc, line_sigma, fill_sigma, info_text

ani = FuncAnimation(fig, update, frames=range(500), interval=50, blit=False)
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()
