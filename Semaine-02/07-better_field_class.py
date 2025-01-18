import matplotlib.pyplot as plt 
import numpy as np


"""
Nous voulons continuer d'améliorer la classe pour faciliter son usage et surtout la rendre plus flexible.
Nous allons créer des méthodes qui vont nous permettre d'éviter de modifier les variables directement
et nous allons tenter d'ajouter une façon de mieux utiliser le graphique.

Pour ce faire, nous allons ajouter des méthodes qui modifient les variables, et qui nous aident à créer
des champs ou des coordonnées.

Nous gardons aussi la figure qui est créée par plt.quiver() et nous pourrons la modifier si necessaire.
Dans matplotlib, une figure s'appelle des "axes".  Je n'ai pas inventé le nom.

Nous pouvons utiliser le code de facon un peu plus simple:

```
	field = VectorField2D(size=20)
	field.display() # Le défaut est un champ en sinus et cosinus

	X,Y = field.xy_mesh()
	field.assign_field_components(U = X/10, V = Y*Y/100)
	field.display()

```

"""

class VectorField2D:
	def __init__(self, size=None, X=None, Y=None, U=None, V=None):
		if size is not None:
			X, Y = self.create_square_meshgrid(size)
		elif X is None or Y is None:
			X, Y = self.create_square_meshgrid()

		if len(X) != len(Y):
			raise ValueError("Les composantes X et Y doivent avoir le meme nombre d'éléments")

		self.X = X
		self.Y = Y

		if U is None or V is None:
			U, V = self.create_demo_field_components()

		self.U = U
		self.V = V

		self.quiver_axes = None

		self.validate_arrays() # Avant d'aller plus loin, nous voulons un champ valide

	def create_square_meshgrid(self, size=20, N=19):
		x = np.linspace(-size/2,size/2, N)
		y = np.linspace(-size/2,size/2, N)
		return np.meshgrid(x,y)

	def create_demo_field_components(self):
		U = np.sin(self.X/2)
		V = np.cos(self.Y/2)
		return U, V

	def validate_arrays(self):
		if self.X is None or self.Y is None:
			raise ValueError("Les coordonnées X et Y ne sont pas assignées")

		if self.U is None or self.V is None:
			raise ValueError("Les composantes U et V du champ vectoriel ne sont pas assignées en tout point X et Y")

		if len(self.U) != len(self.V):
			raise ValueError("Les composantes U et V doivent avoir le meme nombre d'éléments")

		if len(self.U) != len(self.X):
			raise ValueError("Les composantes U,V doivent avoir le meme nombre d'éléments que X et Y")

	def xy_mesh(self):

		return self.X, self.Y

	def assign_field_components(self, U, V):
		self.U = U
		self.V = V
		self.validate_arrays()

		self.quiver_axes = None # Nous avons changé le champ, la figure n'est plus valide

	def display(self):
		self.validate_arrays()

		if self.quiver_axes is None:
			self.quiver_axes = plt.subplot(1,1,1)
			self.quiver_axes.tick_params(direction="in")

			lengths = np.sqrt(self.U*self.U+self.V*self.V)
			self.quiver_axes = plt.quiver(self.X, self.Y, self.U, self.V, lengths, units='xy', scale=1, width=0.15)
			plt.show() 



if __name__ == "__main__":
	field = VectorField2D(size=20)
	field.display() # Le défaut est un champ en sinus et cosinus

	X,Y = field.xy_mesh()
	field.assign_field_components(U = X/10, V = Y*Y/100)
	field.display()

