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
random.seed(42)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TrentoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        unlabelled_size=1000,
        do_preprocess=True,
        # **kwargs
    ) -> None:
        super().__init__()

        image_hyper = torch.tensor(tifffile.imread(data_dir+"hyper_Italy.tif")) # [63,166,600]
        image_lidar = torch.tensor(tifffile.imread(data_dir+"LiDAR_Italy.tif")) # [2,166,600]
        x = torch.cat((image_hyper,image_lidar), dim = 0) # [65,166,600]
        x_all = x
        x_all = x_all.reshape(len(x_all),-1)
        x_all = torch.transpose(x_all, 1,0) # [99600,65]
        
        #Normalize to [0,1]
        if do_preprocess: # TODO: Something more sophisticated?
            logger.info("Normalize to 0,1")
            x_min = x_all.min(dim=0)[0] # [65]
            x_max = x_all.max(dim=0)[0] # [65]
            x_all = (x_all- x_min)/(x_max-x_min)
            assert torch.unique(x_all.min(dim=0)[0] == 0.)
            assert torch.unique(x_all.max(dim=0)[0] == 1.)

        y = torch.tensor(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"], dtype = torch.int64) # [166,600] 0 to 6
        y_train_labelled = torch.tensor(io.loadmat(data_dir+"TNsecSUBS_Train.mat")["TNsecSUBS_Train"], dtype = torch.int64) # [166,600] 0 to 6
        y_test = y-y_train_labelled

        y_all = y
        y_all = y_all.reshape(-1) # [99600]
        y_train_labelled = y_train_labelled.reshape(-1) # [99600]
        y_test = y_test.reshape(-1) # [99600]

        train_labelled_indeces = (y_train_labelled!=0)
        x_train_labelled = x_all[train_labelled_indeces] # [819, 65]
        y_train_labelled = y_all[train_labelled_indeces]  # [819]

        unlabelled_indeces = (y_all==0)
        x_unlabelled = x_all[unlabelled_indeces] # []
        y_unlabelled = y_all[unlabelled_indeces] # []
        x_train_unlabelled, _, y_train_unlabelled,_ = train_test_split(x_unlabelled,y_unlabelled,train_size = unlabelled_size)

        x_train = torch.cat((x_train_labelled,x_train_unlabelled), dim=0)
        y_train = torch.cat((y_train_labelled,y_train_unlabelled), dim=0)

        test_indeces = (y_test!=0)
        x_test = x_all[test_indeces] # [29595, 65]
        y_test = y_all[test_indeces]  # [29595]

        plt.figure()
        plt.suptitle('Distribution Lidar pixel values')
        plt.subplot(211)
        for label,name in zip([1,5],["A. Trees", 'Vineyards']):
            label_ind = np.where(y == label)[0]
            hist_values = x_all[label_ind, -1]
            histogram, bin_edges = np.histogram(hist_values, bins=100, range=(0, 0.2))
            plt.plot(bin_edges[:-1], histogram, label = name)
            plt.title("Channel 1", fontsize = 6)
            plt.xlabel("normalized pixel value")
            plt.ylabel("number of pixels")
            plt.legend()
        plt.subplot(212)
        for label,name in zip([1,5],["A. Trees", 'Vineyards']):
            label_ind = np.where(y == label)[0]
            hist_values = x_all[label_ind, -2]
            histogram, bin_edges = np.histogram(hist_values, bins=100, range=(0, 0.2))
            plt.plot(bin_edges[:-1], histogram, label = name)
            plt.title("Channel 2", fontsize = 6)
            plt.xlabel("normalized pixel value")
            plt.ylabel("number of pixels")
            plt.legend()
        plt.savefig("images/lidar_distribution.png")

        x_test_labelled, _, y_test_labelled, _ = train_test_split(x_test, y_test, train_size= 0.9, stratify = y_test)

        self.labelled_fraction = len(y_train_labelled)/len(y_train)
        self.train_dataset = TensorDataset(x_train, y_train-1) # 0 to 5
        self.train_dataset_labelled = TensorDataset(x_train_labelled, y_train_labelled-1) # 0 to 5
        self.test_dataset = TensorDataset(x_test, y_test-1) # 0 to 5
        self.test_dataset_labelled = TensorDataset(x_test_labelled, y_test_labelled-1) # 0 to 5
        self.full_dataset = TensorDataset(x_all, y_all) # 0 to 6


# DATASET = TrentoDataset(
#     data_dir = "/Users/plo026/data/Trento/",
# )
# x,y = DATASET.train_dataset.tensors # 819
# print(x.shape, y.shape, torch.unique(y))

# x,y = DATASET.train_dataset_labelled.tensors # 409 
# print(x.shape, y.shape, torch.unique(y))

# x,y = DATASET.test_dataset.tensors # 15107
# print(x.shape, y.shape, torch.unique(y))

# x,y = DATASET.test_dataset_labelled.tensors # 15107
# print(x.shape, y.shape, torch.unique(y))

# x,y = DATASET.full_dataset.tensors # 15107
# print(x.shape, y.shape, torch.unique(y))