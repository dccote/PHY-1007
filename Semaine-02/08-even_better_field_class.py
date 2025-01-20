import matplotlib.pyplot as plt 
import numpy as np


"""
Vous pouvez arreter a tout moment de lire les améliorations pour la classe si vous les trouvez trop 
difficile à comprendre.

J'aimerais pouvoir rapidement changer le champ de facon efficace:

field = VectorField()
field.create_single_charge_field_components()

On remarquera que le champ d'une charge varie rapidement, donc l'utilisation de la longueur de la fleche
pour représenter le champ devient problématique car c'est parfois trop long, ou c'est essentiellement nul
et on ne voit rien. J'ajoute une option pour voir le logarithme de la longueur.
"""

class VectorField2D:
	def __init__(self, size=None, N=19, X=None, Y=None, U=None, V=None):
		if size is not None:
			X, Y = self.create_square_meshgrid(size, N)
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

	@property
	def xy_mesh(self):
		return self.X, self.Y

	@property
	def rphi_mesh(self):
		X,Y = self.xy_mesh
		return np.sqrt(X*X+Y*Y), np.atan2(Y, X)

	@property
	def field_magnitude(self):
		return np.sqrt(self.U*self.U+self.V*self.V)

	@property
	def field_log_magnitude(self):
		return np.log(self.field_magnitude)

	def create_square_meshgrid(self, size=20, N=19):
		x = np.linspace(-size/2,size/2, N)
		y = np.linspace(-size/2,size/2, N)
		return np.meshgrid(x,y)

	def create_null_field_components(self):		
		X,Y = self.xy_mesh
		return X*0, X*0 # C'est un truc pour avoir rapidement une liste de la meme longueur avec des zeros

	def create_demo_field_components(self):
		X,Y = self.xy_mesh
		return np.sin(X/2), np.cos(Y/2)

	def create_single_charge_field_components(self):

		# On pourra se retrouver, parfois, avec R==0 (a l'origin), et donc 1/
		# (R*R) sera infini. On veut éviter les infinités et les
		# discontinuités.  Pour l'instant, pour simplifer, je vais simplement
		# ajouter un tout petit 0.01 a R*R pour eviter que cela donne une
		# division par zero. C'est affreux, mais ca évite les
		# problèmes.
		R, PHI = self.rphi_mesh

		return np.cos(PHI)/np.log(R*R+0.01), np.sin(PHI)/np.log(R*R+0.01)

	def validate_arrays(self):
		if self.X is None or self.Y is None:
			raise ValueError("Les coordonnées X et Y ne sont pas assignées")

		if self.U is None or self.V is None:
			raise ValueError("Les composantes U et V du champ vectoriel ne sont pas assignées en tout point X et Y")

		if len(self.U) != len(self.V):
			raise ValueError("Les composantes U et V doivent avoir le meme nombre d'éléments")

		if len(self.U) != len(self.X):
			raise ValueError("Les composantes U,V doivent avoir le meme nombre d'éléments que X et Y")

	def assign_field_components(self, U, V):
		self.U = U
		self.V = V
		self.validate_arrays()

		self.quiver_axes = None # Nous avons changé le champ, la figure n'est plus valide

	def display(self, is_color=True):
		self.validate_arrays()


		if self.quiver_axes is None:
			self.quiver_axes = plt.subplot(1,1,1)
			self.quiver_axes.tick_params(direction="in")

		self.quiver_axes.cla()

		if is_color:
			lengths = self.field_magnitude
			lengths /= np.max(lengths)
			self.quiver_axes.quiver(self.X, self.Y, self.U, self.V, lengths)
		else:
			self.quiver_axes.quiver(self.X, self.Y, self.U, self.V)

		plt.show()
		self.quiver_axes = None




if __name__ == "__main__": # C'est la façon rigoureuse d'ajouter du code après une classe
	field = VectorField2D(size=20, N=9)
	field.display() # Le défaut est un champ en sinus et cosinus

	U, V = field.create_single_charge_field_components()
	field.assign_field_components(U, V)
	field.display(is_color=True)
	field.display(is_color=False)

