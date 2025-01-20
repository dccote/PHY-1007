import matplotlib.pyplot as plt 
import numpy as np


"""
La classe précédente réussit à refaire ce que nous faisions avant, mais 
isole un peu les détails des coordonnées (i.e. le domain 2D) avec lesquelles 
nous faisons les calculs.

L'idée derrière cette stratégie permet, par exemple, de simplement demander
les xy_mesh a la classe, mais nous pouvons par exemple demander les valeurs
par rapport à une autre origine pour une charge placé a (xo, yo) i.e.
ailleurs qu'à l'origine.

Maintenant, il devient plus simple de transformer un peu le fonctionnement: la
classe VectorField2D est responsable de calculer des champs: elle "connnait
l'electromagnetisme", nous allons donc lui demander le champ d'une charge placée
a (xo, yo) et l'ajouter au champ total.

1. Donc on commence avec un champ nul
2. La fonction assign_field_components a été renommée add_field_components
3. On ajoute des charges a une position (xo, yo) avec une charge q (qui peut etre negative)
4. On en ajoute tant qu'on veut et on visualise

"""

class SurfaceDomain:
	def __init__(self, size=20, N=19, X=None, Y=None):
		if size is not None:
			X, Y = self.create_square_meshgrid(size, N)
		elif X is None or Y is None:
			X, Y = self.create_square_meshgrid()

		if len(X) != len(Y):
			raise ValueError("Les composantes X et Y doivent avoir le meme nombre d'éléments")

		self._X = X # Lorsqu'on utilise _ devant une variable, c'est une convention de ne pas l'appeler directement
		self._Y = Y # et d'utiliser les @property accessors

	def xy_mesh(self, xo=0, yo=0):
		"""

		"""
		return self._X-xo, self._Y-yo

	def rphi_mesh(self, xo=0, yo=0):
		X,Y = self.xy_mesh(xo, yo)
		return np.sqrt(X*X+Y*Y), np.atan2(Y, X)

	def create_square_meshgrid(self, size=20, N=19):
		x = np.linspace(-size/2,size/2, N)
		y = np.linspace(-size/2,size/2, N)
		return np.meshgrid(x,y)

class VectorField2D:
	def __init__(self, surface=None, U=None, V=None):
		if surface is None:
			surface = SurfaceDomain()

		self.domain = surface

		if U is None or V is None:
			U, V = self.create_null_field_components()

		self.U = U
		self.V = V

		self.quiver_axes = None

		self.validate_arrays() # Avant d'aller plus loin, nous voulons un champ valide


	@property
	def field_magnitude(self):
		return np.sqrt(self.U*self.U+self.V*self.V)

	def create_null_field_components(self):		
		X,Y = self.domain.xy_mesh()
		return X*0, X*0 # C'est un truc pour avoir rapidement une liste de la meme longueur avec des zeros

	def create_demo_field_components(self):
		X,Y = self.domain.xy_mesh()
		return np.sin(X/2), np.cos(Y/2)

	def create_single_charge_field_components(self, xo=0, yo=0, q=1):

		# On pourra se retrouver, parfois, avec R==0 (a l'origin), et donc 1/(R*R) 
		# sera infini. On veut éviter les infinités et les discontinuités.
		# Pour l'instant, pour simplifer, je vais simplement ajouter un tout
		# petit 0.01 a R*R pour eviter que cela donne une division par zero.
		# C'est affreux, mais ca évite les problèmes.
		R, PHI = self.domain.rphi_mesh(xo, yo)

		return q*np.cos(PHI)/(R*R+0.01), q*np.sin(PHI)/(R*R+0.01)

	def validate_arrays(self):
		if self.U is None or self.V is None:
			raise ValueError("Les composantes U et V du champ vectoriel ne sont pas assignées en tout point X et Y")

		if len(self.U) != len(self.V):
			raise ValueError("Les composantes U et V doivent avoir le meme nombre d'éléments")

	def add_field_components(self, U, V):
		self.U += U
		self.V += V
		self.validate_arrays()

		self.quiver_axes = None # Nous avons changé le champ, la figure n'est plus valide

	def display(self, use_color=True, **quiver_kwargs):
		self.validate_arrays()

		if self.quiver_axes is None:
			self.quiver_axes = plt.subplot(1,1,1)
			self.quiver_axes.tick_params(direction="in")

		self.quiver_axes.cla()

		X,Y = self.domain.xy_mesh()

		if use_color:
			lengths = self.field_magnitude
			self.quiver_axes.quiver(X, Y, self.U/lengths, self.V/lengths, lengths, units='xy', width=0.10, *quiver_kwargs)
		else:
			self.quiver_axes.quiver(X, Y, self.U, self.V, units='xy', width=0.05, *quiver_kwargs)

		plt.show()
		self.quiver_axes = None




if __name__ == "__main__": # C'est la façon rigoureuse d'ajouter du code après une classe
	domain=SurfaceDomain(size=20, N=19)
	field = VectorField2D(surface=domain) # Le domaine par défaut est -10 a 10 avec 19 points par dimension

	U, V = field.create_single_charge_field_components(xo=3, yo=0, q=1)
	field.add_field_components(U, V)
	U, V = field.create_single_charge_field_components(xo=-3, yo=0, q=1)
	field.add_field_components(U, V)
	field.display(use_color=True)

