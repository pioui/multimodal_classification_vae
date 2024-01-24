"""
This Python script creates the grayscale representation from the Lidar information for Houston and Trento dataset

Usage:
  python3 scripts/print_lidar.py 

"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from mcvae.utils import normalize

percentile = 97

data_dir = "/home/pigi/data/houston/"
images_dir =  "outputs/houston/images/"
image_lidar = np.array(tifffile.imread(data_dir+"houston_lidar.tif")) # [7,  1202, 4768]
dataset = "houston"

for i in range(7):
    X_lidar = image_lidar[i,:]
    X_lidar = X_lidar.reshape(-1)
    print(X_lidar.shape)
    print(i, X_lidar.max(), X_lidar.min())

    value = np.percentile(X_lidar, percentile)
    for j in range(len(X_lidar)):
        if X_lidar[j] > value:
            X_lidar[j]= value
    print(i, X_lidar.max(), X_lidar.min())

    x_min = X_lidar.min()
    x_max = X_lidar.max()
    X_lidar = (X_lidar- x_min)/(x_max-x_min)
    print(i, X_lidar.max(), X_lidar.min())

    if i==0: X_r = X_lidar
    if i==1: X_g = X_lidar
    if i==2: X_b = X_lidar

    X_lidar = X_lidar.reshape(1202, 4768)
    plt.imshow(X_lidar, interpolation='nearest', 
    vmin=0, vmax=1, 
    cmap='gray'
    )
    plt.axis('off')
    plt.savefig(f"{images_dir}{dataset}_Lidar_{i}.png",bbox_inches='tight', pad_inches=0, dpi=500)


X_rgb = np.stack((X_r.reshape(1202, 4768), X_g.reshape(1202, 4768), X_b.reshape(1202, 4768)), axis = 2)
plt.imshow(X_rgb, interpolation='nearest', 
vmin=0, vmax=1, 
)
plt.axis('off')
plt.savefig(f"{images_dir}{dataset}_RGB_LiDAR.png",bbox_inches='tight', pad_inches=0, dpi=500)

data_dir = "/home/pigi/data/trento/"
images_dir =  "outputs/trento/images/"
image_lidar = np.array((tifffile.imread(data_dir+"LiDAR_Italy.tif"))) # [2,  1166,600]
dataset = "trento"

for i in range(2):
    X_lidar = image_lidar[i,:]
    X_lidar = X_lidar.reshape(-1)
    print(X_lidar.shape)
    print(i, X_lidar.max(), X_lidar.min())

    value = np.percentile(X_lidar, percentile)
    for j in range(len(X_lidar)):
        if X_lidar[j] > value:
            X_lidar[j]= value
    print(i, X_lidar.max(), X_lidar.min())

    x_min = X_lidar.min()
    x_max = X_lidar.max()
    X_lidar = (X_lidar- x_min)/(x_max-x_min)
    print(i, X_lidar.max(), X_lidar.min())

    X_lidar = X_lidar.reshape(166,600)
    plt.imshow(X_lidar, interpolation='nearest', 
    vmin=0, vmax=1, 
    cmap='gray'
    )
    plt.axis('off')
    plt.savefig(f"{images_dir}{dataset}_Lidar_{i}.png",bbox_inches='tight', pad_inches=0, dpi=500)