import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1,1,9) # 9 valeurs différentes entre -1 et 1
y = np.linspace(-1,1,9) # 9 valeurs différentes entre -1 et 1
X,Y = np.meshgrid(x,y)  # Les 81 combinaisons possibles pour former la grille


U = [ 2, 2, 2, 3, 3, 3, 4, 4, 4] * 9 # Je sauve du temps et je repete 9 fois la meme liste
V = [ 0 ] * 81

plt.quiver(X, Y, U, V)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.grid()
plt.show()