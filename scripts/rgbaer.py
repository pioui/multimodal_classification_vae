import numpy as np
import matplotlib.pyplot as plt

from mcvae.utils import generate_latex_matrix_from_dict
from houston_config import images_dir
x = np.load("/home/pigi/repos/multimodal_classification_vae/outputs/houston_H.npy")

x_min = x.min() # [57]
x_max = x.max() # [57]
H = (x- x_min)/(x_max-x_min)


print(np.max(H))
latex_code = ''
j=0
for row in H:


    latex_code += "$ "
    latex_code +=  "$ & $".join(["{:.2e}".format(i) for i in row]) 
    latex_code += "$ & " + f"$ c_{'{'}{j}{'}'}$" + " \\\\\n"
    j=j+1

print(latex_code)



# Create a sample NumPy array (replace this with your actual data)

data = np.log10(H)

tick_labels = [r'$c_{%d}$' % (i + 1) for i in range(20)]

# # Set custom tick labels for both axes
# plt.xticks(range(20), tick_labels, rotation=90)  # Rotation for x-axis labels
# plt.yticks(range(20), tick_labels)


# Set custom tick labels for both axes


# Create the heatmap using Matplotlib's imshow
plt.figure(figsize = (8,8))
plt.tick_params(axis='x', top=True, labeltop=True, bottom = False, labelbottom = False)
plt.imshow(data, interpolation='nearest', cmap='gnuplot')  # 'viridis' is just one of many available colormaps
# plt.xticks(np.arange(data.shape[1]), labels=np.arange(data.shape[1]))
# plt.yticks(np.arange(data.shape[0]), labels=np.arange(data.shape[0]))

plt.xticks(range(len(tick_labels)), tick_labels)
plt.yticks(range(len(tick_labels)), tick_labels)

plt.colorbar(shrink = 0.8)  # Add a colorbar to the plot for reference
plt.savefig(f"{images_dir}H_heatmap.png",bbox_inches='tight', pad_inches=0, dpi=500)
