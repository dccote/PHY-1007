import matplotlib.pyplot as plt
import numpy as np

N = 19

x = np.linspace(-10,10,N) # N valeurs différentes entre -1 et 1
y = np.linspace(-10,10,N) # N valeurs différentes entre -1 et 1
X,Y = np.meshgrid(x,y)  # Les 81 combinaisons possibles pour former la grille

U = np.sin(X/2) # La composante en x des vecteurs
V = np.cos(Y/2) # La composante en y des vecteurs

lengths = np.sqrt(U*U+V*V) # Le 5e argument donne la couleur selon la colormap

plt.tick_params(direction="in")

plt.quiver(X, Y, U, V, lengths) # On peut changer la couleur avec la longueur
#plt.quiver(X, Y, U, V, lengths, cmap=cm.inferno) # Si vous êtes un artiste
#plt.quiver(X, Y, U, V, lengths, cmap=cm.jet) # Si vous voulez choquer Louis

plt.show()