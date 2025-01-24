import matplotlib.pyplot as plt
import numpy as np

N = 19

x = np.linspace(-10,10,N) # N valeurs différentes entre -1 et 1
y = np.linspace(-10,10,N) # N valeurs différentes entre -1 et 1
X,Y = np.meshgrid(x,y)  # Les 81 combinaisons possibles pour former la grille

R = np.sqrt(X*X+Y*Y)
PHI = np.arctan2(Y, X)

# On pourra se retrouver, parfois, avec R==0 (a l'origin), et donc
# 1/(R*R) sera infini. On veut éviter les infinités et les
# discontinuités.  Pour l'instant, pour simplifer, je vais simplement
# ajouter un tout petit 0.01 a R*R pour eviter que cela donne une
# division par zero. C'est affreux, mais ca évite les
# problèmes.

U = np.cos(PHI)/(R*R+0.01)
V = np.sin(PHI)/(R*R+0.01)

lengths = np.sqrt(U*U+V*V) # Le 5e argument donne la couleur selon la colormap

plt.quiver(X, Y, U, V, lengths) # On peut changer la couleur avec la longueur
#plt.quiver(X, Y, U, V, lengths, cmap=cm.inferno) # Si vous êtes un artiste
#plt.quiver(X, Y, U, V, lengths, cmap=cm.jet) # Si vous voulez choquer Louis

plt.tick_params(direction="in")
plt.show()