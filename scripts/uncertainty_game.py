from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def centroid(p1, p2):
    return 1-np.sqrt((2/3-p1-p2)**2+ (p1-1/3)**2 + (p2-1/3)**2)/(2/3)

def variance(p1, p2):
    return ((1-p1-p2)+ 4*p1 + 9*p2 - (1-p1-p2+ 2*p1 + 3*p2)**2) / (8/12)

p1 = np.linspace(0, 1, 100)
p2 = np.linspace(0, 1, 100)

# X_, Y_ = np.meshgrid(p1, p2)
# X = X_[X_+Y_ <=1 ]
# Y = Y_[X_+Y_ <=1 ]

X, Y = np.meshgrid(p1, p2)

print()
print("Centroid ")
Z = centroid(X, Y)
print("Max: ",np.max(Z))
ind = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
print(X[ind], Y[ind])
print("1,0,0 :",centroid(0,0))
print("0,1,0 :",centroid(1,0))
print("0,0,1 :",centroid(0,1))
print("1/3,1/3,1/3 :",centroid(1/3,1/3))

print("Min: ", np.min(Z))
ind = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
print(X[ind], Y[ind])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 100, cmap='binary')
ax.set_xlabel('p1')
ax.set_ylabel('p2')
ax.set_zlabel('Uncertainty')

# plt.show()
print()
print("Variance ")
Z = variance(X, Y)
print("Max: ",np.max(Z))
ind = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
print(X[ind], Y[ind])

print("Min: ", np.min(Z))
ind = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
print(X[ind], Y[ind])
print("1,0,0 :",variance(0,0))
print("0,1,0 :",variance(1,0))
print("0,0,1 :",variance(0,1))
print("1/3,1/3,1/3 :",variance(1/3,1/3))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 100, cmap='binary')
ax.set_xlabel('p1')
ax.set_ylabel('p2')
ax.set_zlabel('Uncertainty')

plt.show()


