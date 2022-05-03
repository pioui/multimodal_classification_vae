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


class trentoMultimodalDataset(Dataset):
    def __init__(
        self,
        data_dir,
        samples_per_class=200,
        train_size=0.5,
        do_preprocess=True,
    ) -> None:
        super().__init__()

        image_hyper = torch.tensor(tifffile.imread(data_dir+"hyper_Italy.tif")) # [63,166,600]
        image_lidar = torch.tensor(tifffile.imread(data_dir+"LiDAR_Italy.tif")) # [2,166,600]
        x = torch.cat((image_hyper,image_lidar), dim = 0) # [65,166,600]
        x_all = x
        x_all = x_all.reshape(len(x_all),-1)
        x_all = torch.transpose(x_all, 1,0) # [99600,65]
        
        #Normalize to [0,1]
        if do_preprocess: 
            x_all = normalize(x_all).float()

        y = torch.tensor(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"], dtype = torch.int64) # [166,600] 0 to 6
        y_all = y
        y_all = y_all.reshape(-1) # [99600]

        train_inds = []
        for label in y_all.unique():
            label_ind = np.where(y_all == label)[0]
            samples = samples_per_class
            if label == 0:
                labelled_exs = np.random.choice(label_ind, size=(len(y_all.unique())-1)*samples, replace=False)
            else:
                while (len(label_ind)< samples) : samples = int(samples/2)
                labelled_exs = np.random.choice(label_ind, size=samples_per_class, replace=False)
            train_inds.append(labelled_exs)
        train_inds = np.concatenate(train_inds)

        x_all_train = x_all[train_inds]
        y_all_train = y_all[train_inds]
        
        x_train, x_test, y_train, y_test = train_test_split(
            x_all_train, y_all_train, train_size = train_size, random_state = 42, stratify = y_all_train
        ) # 0 to 20


        train_labelled_indeces = (y_train!=0)
        x_train_labelled = x_train[train_labelled_indeces] # [787260,57]
        y_train_labelled = y_train[train_labelled_indeces] # [787260] 1 to 20, 255

        test_labelled_indeces = (y_test!=0)
        x_test_labelled = x_test[test_labelled_indeces] # [787260,57]
        y_test_labelled = y_test[test_labelled_indeces] # [787260] 1 to 20, 255

        self.labelled_fraction = len(y_train_labelled)/len(y_train)
        self.train_dataset = TensorDataset(x_train[:,:63],x_train[:,63:], y_train-1) # 0 to 5
        self.train_dataset_labelled = TensorDataset(x_train_labelled[:,:63],x_train_labelled[:,63:], y_train_labelled-1) # 0 to 5
        self.test_dataset = TensorDataset(x_test[:,:63],x_test[:,63:], y_test-1) # 0 to 5
        self.test_dataset_labelled = TensorDataset(x_test_labelled[:,:63],x_test_labelled[:,63:], y_test_labelled-1) # 0 to 5
        self.full_dataset = TensorDataset(x_all[:,:63],x_all[:,63:], y_all) # 0 to 6
        log_train_test_split([y_all, y_train, y_train_labelled, y_test, y_test_labelled])

if __name__ == "__main__":

    DATASET = trentoMultimodalDataset(
        data_dir = "/home/plo026/data/trento/",
    )
    x1,x2,y = DATASET.train_dataset.tensors # 1819, -1 to 5
    print(x1.shape, x2.shape, y.shape, torch.unique(y))

    x1,x2,y = DATASET.train_dataset_labelled.tensors # 819 0 to 5
    print(x1.shape, x2.shape, y.shape, torch.unique(y))

    x1,x2,y = DATASET.test_dataset.tensors # 29395, 0 to 5
    print(x1.shape, x2.shape, y.shape, torch.unique(y))

    x1,x2,y = DATASET.test_dataset_labelled.tensors # 26455, 0 to 5
    print(x1.shape, x2.shape, y.shape, torch.unique(y))

    x1,x2,y = DATASET.full_dataset.tensors # 99600, 0 to 6
    print(x1.shape, x2.shape, y.shape, torch.unique(y))
