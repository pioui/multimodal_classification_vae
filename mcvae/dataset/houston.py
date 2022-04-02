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

from mcvae.utils import normalize

random.seed(42)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class HoustonDataset(Dataset):
    def __init__(
        self,
        data_dir,
        total_size = 5000, 
        train_size=0.5,
        do_preprocess=True,
        # **kwargs
    ) -> None:
        super().__init__()

        image_hyper = torch.tensor(tifffile.imread(data_dir+"houston_hyper.tif")) # [50,1202,4768]
        image_lidar = torch.tensor(tifffile.imread(data_dir+"houston_lidar.tif")) # [7,1202,4768]
        x = torch.cat((image_hyper,image_lidar), dim = 0) # [57,1202,4768]
        y = torch.tensor(tifffile.imread(data_dir+"houston_gt.tif"), dtype = torch.int64) # [1202,4768]

        x_all = x
        x_all = x_all.reshape(len(x_all),-1) # [57,5731136]
        x_all = torch.transpose(x_all, 1,0) # [5731136,57]
        if do_preprocess: 
            x_all = normalize(x_all).float()
        y_all = y
        y_all = y_all.reshape(-1) # [5731136] 0 to 20
        
        x_train,_, y_train,_ = train_test_split(
            x_all, y_all, train_size = total_size, stratify = y_all
            )

        x_train, x_test, y_train, y_test = train_test_split(
            x_train, y_train, train_size = train_size, random_state = 42, stratify = y_train
            ) # 0 to 20

        train_labelled_indeces = (y_train!=0)
        x_train_labelled = x_train[train_labelled_indeces] # [787260,57]
        y_train_labelled = y_train[train_labelled_indeces] # [787260] 0 to 18, 255

        test_labelled_indeces = (y_test!=0)
        x_test_labelled = x_test[test_labelled_indeces] # [787260,57]
        y_test_labelled = y_test[test_labelled_indeces] # [787260] 0 to 18, 255

        self.labelled_fraction = len(y_train_labelled)/len(y_train)
        self.train_dataset = TensorDataset(x_train, y_train-1) # 0 to 5
        self.train_dataset_labelled = TensorDataset(x_train_labelled, y_train_labelled-1) # 0 to 5
        self.test_dataset = TensorDataset(x_test, y_test-1) # 0 to 5
        self.test_dataset_labelled = TensorDataset(x_test_labelled, y_test_labelled-1) # 0 to 5
        self.full_dataset = TensorDataset(x_all, y_all) # 0 to 6


# DATASET = HoustonDataset(
#     data_dir = "/Users/plo026/data/Houston/",
# )

# x,y = DATASET.full_dataset.tensors # [5731136] 0 to 20
# print(x.shape, y.shape, torch.unique(y))

# x,y = DATASET.train_dataset.tensors # [1719340] -1 to 19, 255
# print(x.shape, y.shape, torch.unique(y)) 

# x,y = DATASET.train_dataset_labelled.tensors # [605673] 0 to 19, 255
# print(x.shape, y.shape, torch.unique(y))

# x,y = DATASET.test_dataset.tensors # [4011796] -1 to 19, 255
# print(x.shape, y.shape, torch.unique(y))

# x,y = DATASET.test_dataset_labelled.tensors # [1413237] 0 to 19, 255
# print(x.shape, y.shape, torch.unique(y))

