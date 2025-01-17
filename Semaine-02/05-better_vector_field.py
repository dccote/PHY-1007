import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize


""" La fonction quiver (i.e. en anglais quiver = carquois pour tenir des flèches)
est très simple: on passe les coordonnées x,y et les longueurs u,v des vecteurs que
l'on veut tracer. Chaque valeur est dans une list séparée. 

Pour facilement construire une liste de coordonnées sur une grille, on peut utiliser
la fonction mesh de numpy: on lui donne nos valeurs en X (seulement) et en Y (seulement)
et ensuite la fonction mesh les combinera pour compléter la grille.

Encore mieux, on peut utiliser X et Y (qui sont des arrays numpy) comme des variables.
Ainsi, en ecrivant "sin(X)" on aura le resultat de la fonction pour chaque valeur de la liste.
Les listes U et V auront automatiquement la même longueurs que X et Y, et les éléments
pour chaque indice seront des éléments correspondant.

"""

N = 19

x = np.linspace(-10,10,N) # N valeurs différentes entre -1 et 1
y = np.linspace(-10,10,N) # N valeurs différentes entre -1 et 1
X,Y = np.meshgrid(x,y)  # Les 81 combinaisons possibles pour former la grille

U = np.sin(X/2) # La composante en x des vecteurs 
V = np.cos(Y/2) # La composante en y des vecteurs 

lengths = np.sqrt(U*U+V*V) # Le 5e argument (0 à 1) donne la couleur selon la colormap

plt.tick_params(direction="in")

plt.quiver(X, Y, U, V, lengths, units='xy', scale=1, width=0.15) # On peut changer la couleur avec la longueur
# plt.quiver(X, Y, U, V, lengths, units='xy', cmap=cm.inferno, scale=1, width=0.15) # Si vous êtes un artiste
# plt.quiver(X, Y, U, V, lengths, units='xy', cmap=cm.hsv, scale=1, width=0.15) # Si vous voulez choquer Louis

plt.show() 
