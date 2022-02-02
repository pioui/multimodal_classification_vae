from ctypes import sizeof
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torch
import random

import tifffile
from scipy import io
import numpy as np

from sklearn.model_selection import train_test_split


def pixelwise_reshape(input):
    """
    compress
    """
    if len(input.shape)==3:
        return input.reshape(-1,input.shape[0])
    if len(input.shape)==2:
        return input.reshape(-1)



data_dir = "/home/pigi/Documents/UiT_Internship/Trento/Trento/"
class TrentoDataset(Dataset):
    def __init__(
        self,
        labelled_fraction = 0.9,
        labelled_proportions = 0.2,
        test_size=0.7,
        do_1d=True,
        do_preprocess=True,
    
    ) -> None:
        """
            Definition of trento Dataset loader
            The data are loaded from data_dir
            only labelled data are used (non zero labels)
            the lidar and hyperspectral data are concatenated 
            so for each pixel we have a feature array with 65 elements, 
            the 2 first are lidar and therest hypersoectral

            we keep 90% of the train set labelled and the rest 10% as unlabelled
        """
        super().__init__()

        self.labelled_fraction = labelled_fraction

        image_hyper = pixelwise_reshape(torch.tensor(tifffile.imread(data_dir+"hyper_Italy.tif"))) # 99600,63
        image_lidar = pixelwise_reshape(torch.tensor(tifffile.imread(data_dir+"LiDAR_Italy.tif"))) # 99600,2
        x = torch.cat((image_hyper,image_lidar), dim = 1) # 99600,65

        gt_train= io.loadmat(data_dir+"TNsecSUBS_Train.mat")["TNsecSUBS_Train"] 
        gt_test = io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"] - gt_train
        gt_train = pixelwise_reshape(torch.tensor(gt_train, dtype = torch.int64)) # 99600
        gt_test = pixelwise_reshape(torch.tensor(gt_test, dtype = torch.int64)) # 99600

        train_indeces = (gt_train!=0).nonzero()[:,0] # 819
        test_indeces = (gt_test!=0).nonzero()[:,0] # 29395

        x_train = x[train_indeces] # 819,65
        y_train = torch.tensor(gt_train[train_indeces]) # 819

        x_test = x[test_indeces] # 29395,65
        y_test = torch.tensor(gt_test[test_indeces]) # 29395


        excluded_label = torch.max(torch.unique(y_train))
        labelled_indeces = (y_train!=excluded_label).nonzero()[:,0]
        
        x_train_labelled, x_train_unlabelled, y_train_labelled, y_train_unlabelled = train_test_split(
            x_train[labelled_indeces], y_train[labelled_indeces], train_size = labelled_fraction, stratify = y_train[labelled_indeces]) 

        # print(torch.unique(y_train)-1, y_train.size())
        # print(torch.unique(y_train_labelled)-1, y_train_labelled.size())
        # print(torch.unique(y_test)-1, y_test.size())

        # NOTE: 1 is substracted from the labels so the smaller label is 0, makes things simpler later but will be good to reverse it in the end.

        rdm_inx = torch.randint(0, 29395, (3000,))

        self.train_dataset = TensorDataset(x_train, y_train-1)  #819
        self.train_dataset_labelled = TensorDataset(x_train_labelled, y_train_labelled-1)  #34
        self.test_dataset = TensorDataset(x_test[rdm_inx], y_test[rdm_inx]-1) #29395

        # self.train_dataset = TensorDataset(x_train[:200,:], y_train[:200]-1)  #819
        # self.train_dataset_labelled = TensorDataset(x_train_labelled[:20,:], y_train_labelled[:20]-1)  #34
        # self.test_dataset = TensorDataset(x_test[:300,:], y_test[:300]-1) #29395


# LABELLED_PROPORTIONS = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
# LABELLED_PROPORTIONS = LABELLED_PROPORTIONS / LABELLED_PROPORTIONS.sum()
# LABELLED_FRACTION = 0.05

# DATASET = TrentoDataset(
#     labelled_fraction=LABELLED_FRACTION,
#     labelled_proportions=LABELLED_PROPORTIONS,
#     do_1d=True,
#     test_size=0.5,
# )

# X_TRAIN, Y_TRAIN = DATASET.train_dataset.tensors #No
# print(torch.unique(Y_TRAIN))