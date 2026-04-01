"""
Animation 2D : densité de charge σ(x,y) sur toute la surface d'un conducteur carré.
On part d'une distribution uniforme dans tout le carré.
Le champ E = -∇V pousse les charges (courant J = σ_cond * E).
Les charges migrent vers le bord jusqu'à l'équilibre (σ = 0 à l'intérieur).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paramètres
N_GRID = 200            # résolution de la grille
DEMI_COTE = 1.0
DT = 0.001
Q_TOTAL = 1.0

# Grille 2D
x = np.linspace(-DEMI_COTE * 1.3, DEMI_COTE * 1.3, N_GRID)
y = np.linspace(-DEMI_COTE * 1.3, DEMI_COTE * 1.3, N_GRID)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Masque : intérieur du carré
interieur = (np.abs(X) <= DEMI_COTE) & (np.abs(Y) <= DEMI_COTE)

# Densité de charge initiale : uniforme dans le disque
sigma = np.zeros_like(X)
sigma[interieur] = 1.0
# Normaliser pour charge totale = Q_TOTAL
sigma *= Q_TOTAL / (np.sum(sigma) * dx * dy)

# --- Résolution par diffusion de charge via le champ E ---
# Dans un conducteur, J = conductivité * E, et E = -∇V
# Conservation de la charge : ∂σ/∂t = -∇·J
# Combiné : ∂σ/∂t = conductivité * ∇²V
# Et ∇²V = -σ/ε₀ (Poisson)
# Donc ∂σ/∂t = -(conductivité/ε₀) * σ
# Mais c'est trop simple (décroissance exponentielle).
#
# Approche correcte : à chaque pas,
# 1. Résoudre Poisson ∇²V = -σ/ε₀ pour trouver V
# 2. Calculer E = -∇V
# 3. Faire couler la charge : ∂σ/∂t = -∇·(σ_cond * E)
#
# On utilise une méthode itérative (Jacobi) pour Poisson
# et des différences finies pour le reste.

CONDUCTIVITE = 10.0  # conductivité du matériau
EPSILON_0 = 1.0
N_JACOBI = 200       # itérations Jacobi par pas de temps

def resoudre_poisson_jacobi(sigma, V, n_iter):
    """Résout ∇²V = -σ/ε₀ par itération de Jacobi."""
    rhs = -sigma / EPSILON_0
    for _ in range(n_iter):
        V[1:-1, 1:-1] = 0.25 * (
            V[2:, 1:-1] + V[:-2, 1:-1] +
            V[1:-1, 2:] + V[1:-1, :-2]
            - dx**2 * rhs[1:-1, 1:-1]
        )
        # Condition aux limites : V = 0 loin du disque
        V[0, :] = 0; V[-1, :] = 0
        V[:, 0] = 0; V[:, -1] = 0
    return V

def calculer_champ(V):
    """E = -∇V par différences centrales."""
    Ex = np.zeros_like(V)
    Ey = np.zeros_like(V)
    Ex[1:-1, 1:-1] = -(V[1:-1, 2:] - V[1:-1, :-2]) / (2 * dx)
    Ey[1:-1, 1:-1] = -(V[2:, 1:-1] - V[:-2, 1:-1]) / (2 * dy)
    return Ex, Ey

V = np.zeros_like(sigma)

# --- Visualisation ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# Panneau 1 : densité σ
ax1 = axes[0]
# Masquer l'extérieur du disque
sigma_display = np.where(interieur, sigma, np.nan)
im_sigma = ax1.imshow(sigma_display, extent=[x[0], x[-1], y[0], y[-1]],
                       origin='lower', cmap='hot', vmin=0, vmax=sigma.max() * 2)
carre1 = plt.Rectangle((-DEMI_COTE, -DEMI_COTE), 2*DEMI_COTE, 2*DEMI_COTE, fill=False, color='white', linewidth=2)
ax1.add_patch(carre1)
ax1.set_aspect('equal')
ax1.set_title('Densité σ(x,y)', fontsize=13)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
cb1 = plt.colorbar(im_sigma, ax=ax1, fraction=0.046, pad=0.04)
cb1.set_label('σ')

# Panneau 2 : potentiel V
ax2 = axes[1]
V_display = np.where(interieur, V, np.nan)
im_V = ax2.imshow(V_display, extent=[x[0], x[-1], y[0], y[-1]],
                   origin='lower', cmap='coolwarm', vmin=-0.1, vmax=0.1)
carre2 = plt.Rectangle((-DEMI_COTE, -DEMI_COTE), 2*DEMI_COTE, 2*DEMI_COTE, fill=False, color='black', linewidth=2)
ax2.add_patch(carre2)
ax2.set_aspect('equal')
ax2.set_title('Potentiel V(x,y)', fontsize=13)
ax2.set_xlabel('x')
cb2 = plt.colorbar(im_V, ax=ax2, fraction=0.046, pad=0.04)
cb2.set_label('V')

# Panneau 3 : profil de σ le long de y=0 (coupe horizontale)
ax3 = axes[2]
mid_row = N_GRID // 2
x_inside = x[(x >= -DEMI_COTE) & (x <= DEMI_COTE)]
idx_inside = (x >= -DEMI_COTE) & (x <= DEMI_COTE)
line_profil, = ax3.plot(x_inside, np.zeros_like(x_inside), 'r-', linewidth=2)
fill_profil = ax3.fill_between(x_inside, np.zeros_like(x_inside), alpha=0.3, color='red')
ax3.set_xlabel('x (coupe à y=0)')
ax3.set_ylabel('σ')
ax3.set_title('Profil σ(x, y=0)', fontsize=13)
ax3.set_xlim(-DEMI_COTE, DEMI_COTE)

info_text = fig.text(0.5, 0.02, '', ha='center', fontsize=11, family='monospace')

step_count = [0]

def update(frame):
    global sigma, V, fill_profil

    # Plusieurs sous-pas
    for _ in range(10):
        # 1. Résoudre Poisson
        V = resoudre_poisson_jacobi(sigma, V, N_JACOBI)

        # 2. Calculer E
        Ex, Ey = calculer_champ(V)

        # 3. Courant de charge J = conductivité * E
        Jx = CONDUCTIVITE * Ex
        Jy = CONDUCTIVITE * Ey

        # 4. Conservation : ∂σ/∂t = -∇·J
        div_J = np.zeros_like(sigma)
        div_J[1:-1, 1:-1] = (
            (Jx[1:-1, 2:] - Jx[1:-1, :-2]) / (2 * dx) +
            (Jy[2:, 1:-1] - Jy[:-2, 1:-1]) / (2 * dy)
        )

        sigma -= DT * div_J

        # Garder σ >= 0 et seulement dans le carré
        sigma = np.maximum(sigma, 0.0)
        sigma[~interieur] = 0.0

        # Renormaliser
        Q = np.sum(sigma) * dx * dy
        if Q > 0:
            sigma *= Q_TOTAL / Q

    step_count[0] += 10

    # Mise à jour affichage σ
    sigma_display = np.where(interieur, sigma, np.nan)
    s_max = max(np.nanmax(sigma_display), 1e-10)
    im_sigma.set_data(sigma_display)
    im_sigma.set_clim(0, s_max)

    # Mise à jour affichage V
    V_display = np.where(interieur, V, np.nan)
    v_absmax = max(np.nanmax(np.abs(V_display)), 1e-10)
    im_V.set_data(V_display)
    im_V.set_clim(-v_absmax, v_absmax)

    # Profil coupe horizontale à y=0
    profil = sigma[mid_row, idx_inside]
    line_profil.set_ydata(profil)
    ax3.set_ylim(0, max(profil.max() * 1.2, 1e-10))

    fill_profil.remove()
    fill_profil = ax3.fill_between(x_inside, profil, alpha=0.3, color='red')

    # Info
    centre_mask = (np.abs(X) < DEMI_COTE * 0.3) & (np.abs(Y) < DEMI_COTE * 0.3)
    bord_mask = interieur & ((np.abs(X) > DEMI_COTE * 0.8) | (np.abs(Y) > DEMI_COTE * 0.8))
    sigma_int = np.mean(sigma[centre_mask]) if np.any(centre_mask) else 0
    sigma_bord = np.mean(sigma[bord_mask]) if np.any(bord_mask) else 0
    info_text.set_text(
        f'Pas: {step_count[0]}  |  σ_centre: {sigma_int:.4f}  |  σ_bord: {sigma_bord:.4f}  |  ratio bord/centre: {sigma_bord/max(sigma_int,1e-10):.1f}x'
    )

    return im_sigma, im_V, line_profil, fill_profil, info_text

ani = FuncAnimation(fig, update, frames=range(1000), interval=50, blit=False)
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()
