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

random.seed(42)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class houstonDataset(Dataset):
    def __init__(
        self,
        data_dir,
        samples_per_class=200,
        train_size=0.5,
        do_preprocess=True,
    ) -> None:
        super().__init__()

        image_hyper = torch.tensor(tifffile.imread(data_dir+"houston_hyper.tif")) 
        image_lidar = torch.tensor(tifffile.imread(data_dir+"houston_lidar.tif")) 
        x = torch.cat((image_hyper,image_lidar), dim = 0)                         
        
        y = torch.tensor(tifffile.imread(data_dir+"houston_gt.tif"), dtype = torch.int64) 

        x_all = x
        x_all = x_all.reshape(len(x_all),-1) 
        x_all = torch.transpose(x_all, 1,0)  
        if do_preprocess: 
            x_all = normalize(x_all).float()
        y_all = y
        y_all = y_all.reshape(-1) 
        
        x_train_all, _, y_train_all, _ = train_test_split(
            x_all, y_all, train_size = 40000, random_state = 42, stratify = y_all
        ) 
        x_train, x_test, y_train, y_test = train_test_split(
            x_train_all, y_train_all, train_size = 0.5, random_state = 42, stratify = y_train_all
        ) 

        train_labelled_indeces = (y_train!=0)
        x_train_labelled = x_train[train_labelled_indeces] 
        y_train_labelled = y_train[train_labelled_indeces] 

        test_labelled_indeces = (y_test!=0)
        x_test_labelled = x_test[test_labelled_indeces] 
        y_test_labelled = y_test[test_labelled_indeces] 

        self.labelled_fraction = len(y_train_labelled)/len(y_train)
        self.train_dataset = TensorDataset(x_train, y_train-1)
        self.train_dataset_labelled = TensorDataset(x_train_labelled, y_train_labelled-1) 
        self.test_dataset = TensorDataset(x_test, y_test-1) 
        self.test_dataset_labelled = TensorDataset(x_test_labelled, y_test_labelled-1) 
        self.full_dataset = TensorDataset(x_all, y_all) 
        log_train_test_split([y_all, y_train, y_train_labelled, y_test, y_test_labelled])

if __name__ == "__main__":

    DATASET = houstonDataset(
        data_dir = "/home/pigi/data/houston/",
    )

    x,y = DATASET.full_dataset.tensors 
    print(x.shape, y.shape, torch.unique(y))
    for l in torch.unique(y):
        print(f'Label {l}: {torch.sum(y==l)}')

    x,y = DATASET.train_dataset.tensors 
    print(x.shape, y.shape, torch.unique(y)) 
    for l in torch.unique(y):
        print(f'Label {l}: {torch.sum(y==l)}')

    x,y = DATASET.train_dataset_labelled.tensors
    print(x.shape, y.shape, torch.unique(y))
    for l in torch.unique(y):
        print(f'Label {l}: {torch.sum(y==l)}')

    x,y = DATASET.test_dataset.tensors 
    print(x.shape, y.shape, torch.unique(y))
    for l in torch.unique(y):
        print(f'Label {l}: {torch.sum(y==l)}')

    x,y = DATASET.test_dataset_labelled.tensors 
    print(x.shape, y.shape, torch.unique(y))
    for l in torch.unique(y):
        print(f'Label {l}: {torch.sum(y==l)}')

