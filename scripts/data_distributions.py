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
from houston_config import labels

image_hyper = torch.tensor(tifffile.imread(data_dir+"houston_hyper.tif")) # [50,1202,4768]
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

for label in y_all.unique():
    samples = 100
    label_ind = np.where(y_all == label)[0]
    if label in [2,10,11,12,13,14]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(0,49,50)
        
    fig, ax = plt.subplots(dpi=100)
    ax.plot(y, x_mean, '-')
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, alpha=0.2)
    ax.plot(y, x_mean, 'o', color='tab:purple')
    ax.set_ylim([0, 1])
    ax.grid(True,which='both')
    plt.xlabel('HSI Channels')
    plt.ylabel('Normalized Mean Values')
    plt.savefig(f"outputs/houston_HSI_{labels[label]}.png")

x = image_lidar
y = torch.tensor(tifffile.imread(data_dir+"houston_gt.tif"), dtype = torch.int64) # [1202,4768]

x_all = x
x_all = x_all.reshape(len(x_all),-1) # [57,5731136]
x_all = torch.transpose(x_all, 1,0) # [5731136,57]    
x_all = normalize(x_all).float()

y_all = y
y_all = y_all.reshape(-1) # [5731136] 0 to 20

for label in y_all.unique():
    samples = 100
    label_ind = np.where(y_all == label)[0]
    if label in [2,10,11,12,13,14]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(0,8,7)
        
    fig, ax = plt.subplots(dpi=100)
    ax.plot(y, x_mean, '-')
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, alpha=0.2)
    ax.plot(y, x_mean, 'o', color='tab:purple')
    ax.set_ylim([0, 1])
    ax.grid(True,which='both')
    plt.xlabel('LiDAR Channels')
    plt.ylabel('Normalized Mean Values')
    plt.savefig(f"outputs/houston_LiDAR_{labels[label]}.png")

 
data_dir = "/Users/plo026/data/trento/"
from trento_config import labels

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


for label in y_all.unique():
    samples = 100
    label_ind = np.where(y_all == label)[0]
    if label in [1,2,5]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(0,62,63)
        
    fig, ax = plt.subplots(dpi=100)
    ax.plot(y, x_mean, '-')
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, alpha=0.2)
    ax.plot(y, x_mean, 'o', color='tab:purple')
    ax.set_ylim([0, 1])
    ax.grid(True,which='both')
    plt.xlabel('HSI Channels')
    plt.ylabel('Normalized Mean Values')
    plt.savefig(f"outputs/trento_HSI_{labels[label]}.png")

x = image_lidar
y = torch.tensor(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"], dtype = torch.int64) # [166,600] 0 to 6

x_all = x
x_all = x_all.reshape(len(x_all),-1) # [57,5731136]
x_all = torch.transpose(x_all, 1,0) # [5731136,57]    
x_all = normalize(x_all).float()

y_all = y
y_all = y_all.reshape(-1) # [5731136] 0 to 20


for label in y_all.unique():
    samples = 100
    label_ind = np.where(y_all == label)[0]
    if label in [1,2,5]:
        labelled_exs = np.random.choice(label_ind, size=samples, replace=False)
    else: 
        continue
    x_label = x_all[labelled_exs] #[100,50]
    x_mean = x_label.mean(dim=0) #[50]
    x_var = x_label.std(dim=0) #[50]
    y = np.linspace(0,1,2)
        
    fig, ax = plt.subplots(dpi=100)
    ax.plot(y, x_mean, '-')
    ax.fill_between(y, x_mean - x_var, x_mean + x_var, alpha=0.2)
    ax.plot(y, x_mean, 'o', color='tab:purple')
    ax.set_ylim([0, 0.2])
    ax.grid(True,which='both')
    plt.xlabel('LiDAR Channels')
    plt.ylabel('Normalized Mean Values')
    plt.savefig(f"outputs/trento_LiDAR_{labels[label]}.png")
