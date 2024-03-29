"""
Script to plot the distributions of different HSI and LiDAR channels in Houston data
Usage:
  python3 scripts/houston_distribution.py 
TODO: not very clean code
"""
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

from scipy import io
import tifffile
import numpy as np
import random
from mcvae.utils import normalize

from houston_config import labels, color, images_dir, data_dir

image_hyper = torch.tensor(tifffile.imread(data_dir+"houston_hyper.tif"))[:48] 
image_lidar = torch.tensor(tifffile.imread(data_dir+"houston_lidar.tif")) 

x = image_hyper
y = torch.tensor(tifffile.imread(data_dir+"houston_gt.tif"), dtype = torch.int64) 

x_all = x
x_all = x_all.reshape(len(x_all),-1) 
x_all = torch.transpose(x_all, 1,0) 
x_all = normalize(x_all).float()

y_all = y
y_all = y_all.reshape(-1) 

samples = 5000


fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [4,5,11]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(380,1050,48)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    ax.set_ylim([0, 0.5])

plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_HSI_trees_side.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [6,11]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(380,1050,48)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    ax.set_ylim([0, 0.5])

plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_HSI_earth_side.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [8,9]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(380,1050,48)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    ax.set_ylim([0, 0.5])

plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_HSI_buildings.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [10,11,12,13,14]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(380,1050,48)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    ax.set_ylim([0, 0.5])

plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_HSI_roads.png")


fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [18,19]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(380,1050,48)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    ax.set_ylim([0, 0.5])

plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_HSI_trains_cars.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [10,11,17]:
        labelled_exs = np.random.choice(label_ind, size=500, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(380,1050,48)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    ax.set_ylim([0, 0.5])

plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_HSI_roads_uplots.png")


fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [16,18]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(380,1050,48)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    ax.set_ylim([0, 0.5])

plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_HSI_cars_pplots.png")






y = torch.tensor(tifffile.imread(data_dir+"houston_gt.tif"), dtype = torch.int64) 

percentile = 97
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
    image_lidar[i,:] = X_lidar.reshape((1202, 4768))

x = image_lidar


x_all = x
x_all = x_all.reshape(len(x_all),-1) 
x_all = torch.transpose(x_all, 1,0) 
x_all = normalize(x_all).float()

y_all = y
y_all = y_all.reshape(-1) 

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [4,5,11]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(0,6,7)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o', color = color[label])
    ax.set_ylim([-0.01, 1.3])

plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_LiDAR_trees_side.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [6,11]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(0,6,7)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o', color = color[label])
    ax.set_ylim([-0.01, 1.3])

plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_LiDAR_earth_side.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [8,9]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(0,6,7)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o', color = color[label])
    ax.set_ylim([-0.01, 1.3])

plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_LiDAR_buildings.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [10,11,12,13,14]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(0,6,7)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o', color = color[label])
    ax.set_ylim([-0.01, 1.3])

plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_LiDAR_roads.png")


fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [18,19]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(0,6,7)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o', color = color[label])
    ax.set_ylim([-0.01, 1.3])

plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_LiDAR_trains_cars.png")


fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [10,11,17]:
        labelled_exs = np.random.choice(label_ind, size=500, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(0,6,7)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o', color = color[label])
    ax.set_ylim([-0.01, 1.3])

plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_LiDAR_roads_uplots.png")


fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    label_ind = np.where(y_all == label)[0]
    if label in [16,18]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(0,6,7)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o', color = color[label])
    ax.set_ylim([-0.01, 1.3])

plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_LiDAR_cars_pplots.png")

