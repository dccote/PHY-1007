import matplotlib.pyplot as plt

X = [0,-2]
Y = [0, 1]

U = [4, 7]
V = [2,-3]

plt.quiver(X, Y, U, V)
plt.xlim(-5, 5)
plt.ylim(-5, 5)

plt.grid()
plt.show()