import numpy as np
import torch
import tifffile
from scipy import io
import matplotlib.pyplot as plt
from matplotlib import colors


data_dir = "/Users/plo026/data/trento/"
SHAPE = (166,600)
color = ["black", "red", "gray", "blue", "orange", "green","yellow"]


y = np.array(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"]) # [166,600] 0 to 6
y_true = y.reshape(-1)
image_hyper = torch.tensor(tifffile.imread(data_dir+"hyper_Italy.tif")) # [63,166,600]
x_all = image_hyper.reshape(len(image_hyper),-1)
x_all = torch.transpose(x_all, 1,0)
x_min = x_all.min(dim=0)[0] # [65]
x_max = x_all.max(dim=0)[0] # [65]
x_all = (x_all- x_min)/(x_max-x_min)
hyperrgb = np.zeros((166,600,3))
hyperrgb[:,:,0] = x_all[:,40].reshape(166,600)+0.05
hyperrgb[:,:,1] = x_all[:,20].reshape(166,600)+0.1
hyperrgb[:,:,2] = x_all[:,0].reshape(166,600)+0.2

plt.figure(dpi=1000)
plt.imshow(hyperrgb)
plt.axis('off')
plt.savefig("/Users/plo026/Documents/rgb_trento.png",bbox_inches='tight', dpi=1000, pad_inches=0.0)

plt.figure(dpi=1000)
plt.imshow(y_true.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color))
plt.axis('off')
plt.savefig("/Users/plo026/Documents/gt_trento.png",bbox_inches='tight', dpi=1000, pad_inches=0.0)



data_dir = "/Users/plo026/data/houston/"
color = [
    "black", "limegreen", "lime", "forestgreen", "green", 
    "darkgreen", "saddlebrown", "aqua", "white", 
    "plum",  "red", "darkgray", "dimgray",
    "firebrick", "darkred", "peru", "yellow", "orange",
    "magenta", "blue", "skyblue"
    ]
SHAPE = (1202,4768)

y = np.array(tifffile.imread(data_dir+"houston_gt.tif"), dtype = np.int64) # [1202,4768]
y_true = y.reshape(-1)
image_hyper = torch.tensor(tifffile.imread(data_dir+"houston_hyper.tif")) # [50,1202,4768]
x_all = image_hyper.reshape(len(image_hyper),-1)
x_all = torch.transpose(x_all, 1,0)
x_min = x_all.min(dim=0)[0] # [57]
x_max = x_all.max(dim=0)[0] # [57]
x_all = (x_all- x_min)/(x_max-x_min)
hyperrgb = np.zeros((1202,4768,3))
hyperrgb[:,:,0] = x_all[:,40].reshape(1202,4768)+0.05
hyperrgb[:,:,1] = x_all[:,20].reshape(1202,4768)+0.1
hyperrgb[:,:,2] = x_all[:,0].reshape(1202,4768)+0.2

plt.figure(dpi=1000)
plt.imshow(hyperrgb)
plt.axis('off')
plt.savefig("/Users/plo026/Documents/rgb_houston.png",bbox_inches='tight', dpi=1000, pad_inches=0.0)

plt.figure(dpi=1000)
plt.imshow(y_true.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color))
plt.axis('off')
plt.savefig("/Users/plo026/Documents/gt_houston.png",bbox_inches='tight', dpi=1000, pad_inches=0.0)
