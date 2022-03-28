
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

import numpy as np

labels = ["Unknown", "Apple Trees", "Buildings", "Ground", "Wood", "Vineyard", "Roads"]
color = ["black", "green", "orange", "gold", "blue", "purple","red"]
plt.figure()

plt.subplot(211)
plt.imshow(np.random.randint(1,6,(166,600)), interpolation='nearest', cmap = colors.ListedColormap(color[1:]))
plt.axis('off')
plt.title("Predictions")
plt.subplot(212)
plt.imshow(np.random.randint(0,6,(166,600)), interpolation='nearest', cmap = colors.ListedColormap(color))
plt.axis('off')
plt.title("Ground Truth")

handles = []
for c,l in zip(color, labels):
    handles.append(mpatches.Patch(color=c, label=l))

plt.legend(handles=handles, loc='lower center', prop={'size':10}, bbox_to_anchor=(0.5,-0.55), ncol=4, borderaxespad=0.)
plt.savefig("save.png", bbox_inches='tight')
plt.show()