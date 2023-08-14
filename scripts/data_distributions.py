"""
Script to plot the distributions of different HSI and LiDAR channels in Houston and Trento data
Usage:
  python3 scripts/data_distribution.py 
TODO: not very clean code
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

from scipy import io
import tifffile
import numpy as np
import random
import matplotlib.pyplot as plt
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

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
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
    samples = 300
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
plt.savefig(f"{images_dir}houston_HSI_mix.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [8,9]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(380,1050,48)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label+1])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label+1], alpha=0.2)
    ax.set_ylim([0, 0.5])

plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_HSI_buildings.png")


x = image_lidar
y = torch.tensor(tifffile.imread(data_dir+"houston_gt.tif"), dtype = torch.int64) 

x_all = x
x_all = x_all.reshape(len(x_all),-1) 
x_all = torch.transpose(x_all, 1,0) 
x_all = normalize(x_all).float()

y_all = y
y_all = y_all.reshape(-1) 

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
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
    samples = 300
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
plt.savefig(f"{images_dir}houston_LiDAR_mix.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [8,9]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(0,6,7)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label+1])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label+1], alpha=0.2)
    ax.plot(y, x_mean, 'o', color = color[label+1])
    ax.set_ylim([-0.01, 1.3])

plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}houston_LiDAR_buildings.png")
 
data_dir = "/home/pigi/data/trento/"
from trento_config import labels, color, images_dir, data_dir

image_hyper = torch.tensor(tifffile.imread(data_dir+"hyper_Italy.tif")) 
image_lidar = torch.tensor(tifffile.imread(data_dir+"LiDAR_Italy.tif")) 

x = image_hyper
y = torch.tensor(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"], dtype = torch.int64) 

x_all = x
x_all = x_all.reshape(len(x_all),-1) 
x_all = torch.transpose(x_all, 1,0)  
x_all = normalize(x_all).float()

y_all = y
y_all = y_all.reshape(-1) 
    
fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [1,5]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(402.89,989.09,len(x_var))
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    ax.set_ylim([0, 0.7])

plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}trento_HSI_trees.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [1,2]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(402.89,989.09,len(x_var))
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    ax.set_ylim([0, 0.7])

plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}trento_HSI_mix.png")

x = image_lidar
y = torch.tensor(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"], dtype = torch.int64) 

x_all = x
x_all = x_all.reshape(len(x_all),-1) 
x_all = torch.transpose(x_all, 1,0) 
x_all = normalize(x_all).float()

y_all = y
y_all = y_all.reshape(-1) 


fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [1,5]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(0,1,2)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o',color = color[label])
    ax.set_ylim([-0.01, 0.7])

plt.xticks(y)
plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}trento_LiDAR_trees.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [1,2]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] 
    x_mean = x_label.mean(dim=0) 
    x_var = x_label.std(dim=0) 
    y = np.linspace(0,1,2)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o',color = color[label])
    ax.set_ylim([-0.01, 0.7])

plt.xticks(y)
plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"{images_dir}trento_LiDAR_mix.png")
