import matplotlib.pyplot as plt
import numpy as np


heterophil_matrix = np.array(
    [
        [1,4,4,3,2,4],
        [4,1,4,4,4,3],
        [4,4,1,4,4,3],
        [3,4,4,1,3,4],
        [2,4,4,3,1,4],
        [4,3,3,4,4,1],
    ]
    )

plt.figure(dpi=500)
plt.matshow(heterophil_matrix, cmap="cool")
plt.xticks(np.arange(0,6,1), range(1,7))
plt.yticks(np.arange(0,6,1), range(1,7))
for k in range (len(heterophil_matrix)):
    for l in range(len(heterophil_matrix[k])):
        plt.text(k,l,str(heterophil_matrix[k][l]), va='center', ha='center', fontsize='small') # trento
plt.savefig(f"trento_heterophili.png",bbox_inches='tight', pad_inches=0.2, dpi=500)

heterophil_matrix = np.array(
    [
        [1,2,3,5,5,4,6,6,6,5,5,5,5,5,6,5,5,6,6,6],
        [2,1,3,5,5,4,6,6,6,5,5,5,5,5,6,5,5,6,6,6],
        [3,3,1,5,5,4,6,6,6,5,5,5,5,5,6,5,5,6,6,6],
        [5,5,5,1,2,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],
        [5,5,5,2,1,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],
        [4,4,4,6,6,1,6,6,6,5,5,5,5,5,6,5,2,6,6,6],
        [6,6,6,6,6,6,1,6,6,6,6,6,6,6,6,6,6,6,6,6],
        [6,6,6,6,6,6,6,1,2,5,5,5,5,5,6,5,6,6,6,6],
        [6,6,6,6,6,6,6,2,1,5,5,5,5,5,6,5,6,6,6,6],
        [5,5,5,6,6,5,6,5,5,1,3,3,2,2,6,2,6,6,6,6],
        [5,5,5,6,6,5,6,5,5,3,1,3,3,3,6,4,6,6,6,6],
        [5,5,5,6,6,5,6,5,5,3,3,1,3,3,6,3,6,6,6,6],
        [5,5,5,6,6,5,6,5,5,2,3,3,1,2,6,2,6,6,6,6],
        [5,5,5,6,6,5,6,5,5,2,3,3,2,1,6,2,6,6,6,6],
        [6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,6,6,3,3,6],
        [5,5,5,6,6,5,6,5,5,2,4,3,2,2,6,1,3,6,6,6],
        [5,5,5,6,6,2,6,6,6,6,6,6,6,6,6,3,1,6,6,6],
        [6,6,6,6,6,6,6,6,6,6,6,6,6,6,3,6,6,1,3,6],
        [6,6,6,6,6,6,6,6,6,6,6,6,6,6,3,6,6,3,1,6],
        [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1],
    ]
)


plt.figure(dpi=500)
plt.matshow(heterophil_matrix, cmap="cool")
plt.xticks(np.arange(0,20,1), range(1,21))
plt.yticks(np.arange(0,20,1), range(1,21))
for k in range (len(heterophil_matrix)):
    for l in range(len(heterophil_matrix[k])):
        plt.text(k,l,str(heterophil_matrix[k][l]), va='center', ha='center', fontsize='xx-small') # houston
plt.savefig(f"houston_heterophili.png",bbox_inches='tight', pad_inches=0.2, dpi=500)