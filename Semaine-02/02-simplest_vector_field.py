import matplotlib.pyplot as plt

X = [-1, 0, 1,-1, 0, 1,-1, 0, 1]
Y = [-1,-1,-1, 0, 0, 0, 1, 1, 1]

U = [ 2, 2, 2, 3, 3, 3, 4, 4, 4]
V = [ 0, 0, 0, 0, 0, 0, 0, 0, 0]

plt.quiver(X, Y, U, V)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.grid()
plt.show()