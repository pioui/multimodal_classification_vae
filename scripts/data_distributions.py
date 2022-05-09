import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

from scipy import io
import tifffile
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import random
import matplotlib.pyplot as plt
from mcvae.utils import normalize, log_train_test_split

data_dir = "/Users/plo026/data/houston/"
from houston_config import labels, color

image_hyper = torch.tensor(tifffile.imread(data_dir+"houston_hyper.tif"))[:48] # [50,1202,4768]
image_lidar = torch.tensor(tifffile.imread(data_dir+"houston_lidar.tif")) # [7,1202,4768]

# x = torch.cat((image_hyper,image_lidar), dim = 0) # [57,1202,4768]
x = image_hyper
y = torch.tensor(tifffile.imread(data_dir+"houston_gt.tif"), dtype = torch.int64) # [1202,4768]

x_all = x
x_all = x_all.reshape(len(x_all),-1) # [57,5731136]
x_all = torch.transpose(x_all, 1,0) # [5731136,57]    
x_all = normalize(x_all).float()

y_all = y
y_all = y_all.reshape(-1) # [5731136] 0 to 20

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [10,11,12,13,14]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(380,1050,48)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    # ax.plot(y, x_mean, 'o', color='tab:purple')
    ax.set_ylim([0, 0.5])
    # ax.grid(True,which='both')
plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"outputs/houston_HSI_roads.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [3,13]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(380,1050,48)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    # ax.plot(y, x_mean, 'o', color='tab:purple')
    ax.set_ylim([0, 0.5])
    # ax.grid(True,which='both')
plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"outputs/houston_HSI_mix.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [8,9]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(380,1050,48)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label+1])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label+1], alpha=0.2)
    # ax.plot(y, x_mean, 'o', color='tab:purple')
    ax.set_ylim([0, 0.5])
    # ax.grid(True,which='both')
plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"outputs/houston_HSI_buildings.png")


x = image_lidar
y = torch.tensor(tifffile.imread(data_dir+"houston_gt.tif"), dtype = torch.int64) # [1202,4768]

x_all = x
x_all = x_all.reshape(len(x_all),-1) # [57,5731136]
x_all = torch.transpose(x_all, 1,0) # [5731136,57]    
x_all = normalize(x_all).float()

y_all = y
y_all = y_all.reshape(-1) # [5731136] 0 to 20

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [10,11,12,13,14]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(0,6,7)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o', color = color[label])
    ax.set_ylim([-0.01, 0.7])
    # ax.grid(True,which='both')
plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"outputs/houston_LiDAR_roads.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [3,13]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(0,6,7)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o', color = color[label])
    ax.set_ylim([-0.01, 0.7])
    # ax.grid(True,which='both')
plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"outputs/houston_LiDAR_mix.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [8,9]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(0,6,7)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label+1])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label+1], alpha=0.2)
    ax.plot(y, x_mean, 'o', color = color[label+1])
    ax.set_ylim([-0.01, 1])
    # ax.grid(True,which='both')
plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"outputs/houston_LiDAR_buildings.png")
 
data_dir = "/Users/plo026/data/trento/"
from trento_config import labels,color

image_hyper = torch.tensor(tifffile.imread(data_dir+"hyper_Italy.tif")) # [63,166,600]
image_lidar = torch.tensor(tifffile.imread(data_dir+"LiDAR_Italy.tif")) # [2,166,600]

# x = torch.cat((image_hyper,image_lidar), dim = 0) # [57,1202,4768]
x = image_hyper
y = torch.tensor(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"], dtype = torch.int64) # [166,600] 0 to 6

x_all = x
x_all = x_all.reshape(len(x_all),-1) # [57,5731136]
x_all = torch.transpose(x_all, 1,0) # [5731136,57]    
x_all = normalize(x_all).float()

y_all = y
y_all = y_all.reshape(-1) # [5731136] 0 to 20

    
fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [1,5]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(402.89,989.09,len(x_var))
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    # ax.plot(y, x_mean, 'o', color = color[label])
    ax.set_ylim([0, 0.7])
# plt.grid(which='both')
plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"outputs/trento_HSI_trees.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [1,2]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(402.89,989.09,len(x_var))
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var,color = color[label], alpha=0.2)
    # ax.plot(y, x_mean, 'o', color = color[label])
    ax.set_ylim([0, 0.7])
# plt.grid(which='both')
plt.legend(loc='upper left')
plt.xlabel('Wavelenghth (nm)')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"outputs/trento_HSI_mix.png")



x = image_lidar
y = torch.tensor(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"], dtype = torch.int64) # [166,600] 0 to 6

x_all = x
x_all = x_all.reshape(len(x_all),-1) # [57,5731136]
x_all = torch.transpose(x_all, 1,0) # [5731136,57]    
x_all = normalize(x_all).float()

y_all = y
y_all = y_all.reshape(-1) # [5731136] 0 to 20


fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [1,5]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(0,1,2)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o',color = color[label])
    ax.set_ylim([-0.01, 0.11])
    # ax.grid(True,which='both')
plt.xticks(y)
plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"outputs/trento_LiDAR_trees.png")

fig, ax = plt.subplots(dpi=100)
for label in y_all.unique():
    samples = 300
    label_ind = np.where(y_all == label)[0]
    if label in [1,2]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(0,1,2)
        
    ax.plot(y, x_mean, '-', label = labels[label], color = color[label])
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, color = color[label], alpha=0.2)
    ax.plot(y, x_mean, 'o',color = color[label])
    ax.set_ylim([-0.01, 0.72])
    # ax.grid(True,which='both')
plt.xticks(y)
plt.legend(loc='upper left')
plt.xlabel('LiDAR Channels')
plt.ylabel('Normalized Mean Values')
plt.savefig(f"outputs/trento_LiDAR_mix.png")
