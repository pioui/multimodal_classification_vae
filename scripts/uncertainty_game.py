from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


from mcvae.utils import centroid, variance_heterophil, variance
from trento_config import heterophil_matrix, labels

print('labels:')
for i in range(len(labels)-1):
    print(i+1, labels[i+1])
print()
print("heterophil matrix:")
print(heterophil_matrix)
print()
print("Sure case")
p = [1,0,0,0,0,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))
p = [0,1,0,0,0,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

p = [0,0,1,0,0,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

p = [0,0,0,1,0,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

p = [0,0,0,0,1,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

p = [0,0,0,0,0,1]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))


print()
print("Confusion between 2 classes")
print(" Distance 2")
p = [0.5,0,0,0,0.5,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

p = [0.8,0,0,0,0.2,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))



print(" Distance 3")
p = [0.5,0,0,0.5,0,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

p = [0.8,0,0,0.2,0,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

p = [0,0,0.5,0,0,0.5]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

p = [0,0,0.8,0,0,0.2]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

print(" Distance 4")
p = [0.5,0.5,0,0,0,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

p = [0.8,0.2,0,0,0,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

print()
print("Confusion between more classes")
p = [0.5,0,0,0.2,0.3,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

p = [0.5,0.2,0,0,0.3,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

p = [0.3,0.2,0.1,0.1,0.3,0]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))

p = [1/6,1/6,1/6,1/6,1/6,1/6]
p = np.array([p])
print(p,variance_heterophil(p=p, w=heterophil_matrix))


# heterophil_matrix = np.array([
#     [1,3,2,2],
#     [3,1,2,2],
#     [2,3,1,2],
#     [2,3,2,1],

# ])
# a=3
# b=2
# c=2


# p = [0.51,0.49,0,0]
# p = np.array([p])
# print(p,variance_heterophil(p=p, w=heterophil_matrix))


# p = [0.25,5/12,1/6,1/6]
# p = np.array([p])
# print(p,variance_heterophil(p=p, w=heterophil_matrix))





