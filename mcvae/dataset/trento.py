import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

from scipy import io
import tifffile
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import random

random.seed(42)
logger = logging.getLogger(__name__)


class TrentoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        labelled_fraction,
        labelled_proportions,
        test_size=0.7,
        total_size = 0.17,
        do_1d=False,
        do_preprocess=True,
        # **kwargs
    ) -> None:
        super().__init__()

        assert len(labelled_proportions) == 6
        labelled_proportions = np.array(labelled_proportions)
        assert abs(labelled_proportions.sum() - 1.0) <= 1e-5
        self.labelled_fraction = labelled_fraction
        label_proportions = labelled_fraction * labelled_proportions

        image_hyper = torch.tensor(tifffile.imread(data_dir+"hyper_Italy.tif")) # [63,166,600]
        image_lidar = torch.tensor(tifffile.imread(data_dir+"LiDAR_Italy.tif")) # [2,166,600]
        x = torch.cat((image_hyper,image_lidar), dim = 0) # [65,166,600]
        x_all = x
        x_all = x_all.reshape(len(x_all),-1)
        x_all = torch.transpose(x_all, 1,0)
                #Normalize to [0,1]
        if do_preprocess: # TODO: Something more sophisticated?
            logger.info("Normalize to 0,1")
            x_min = x_all.min(dim=0)[0] # [65]
            x_max = x_all.max(dim=0)[0] # [65]
            x_all = (x_all- x_min)/(x_max-x_min)
            assert torch.unique(x_all.min(dim=0)[0] == 0.)
            assert torch.unique(x_all.max(dim=0)[0] == 1.)
        # # Standarization 
        # #TODO: this can be written (more tidy) as a transform with other preprocessin' when making the dataset 
        # mean = torch.mean(x, dim = 0)
        # std= torch.std(x, dim = 0)
        # x = (x - mean)/std # [99600,65]

        y = torch.tensor(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"], dtype = torch.int64) # [166,600] 0 to 6
        y_all = y-1
        y_all = y_all.reshape(-1)

        valid_indeces = (y!=0)
        xv = x[:,valid_indeces] # [65, 30214]
        
        x = torch.transpose(xv,1,0)
        assert sum(xv[:,0] != x[0,:]) == 0
        y = y[valid_indeces]-1 # [30214] 0 to 5

        #reduce the dataset size to make it easier for my pour cpu
        ind, _ = train_test_split(np.arange(len(x)), train_size=total_size, random_state=42)
        x = x[ind]
        y = y[ind]

        non_labelled = labelled_proportions == 0.0
        assert (
            non_labelled[1:].astype(int) - non_labelled[:-1].astype(int) >= 0
        ).all(), (
            "For convenience please ensure that non labelled numbers are the last ones"
        )
        non_labelled = np.where(labelled_proportions == 0.0)[0]
        if len(non_labelled) >= 1:
            y[np.isin(y, non_labelled)] = int(non_labelled[0])


        # print(torch.unique(y), torch.max(x), torch.min(x), x.shape, y.shape)

        #Normalize to [0,1]
        if do_preprocess: # TODO: Something more sophisticated?
            logger.info("Normalize to 0,1")
            x_min = x.min(dim=0)[0] # [65]
            x_max = x.max(dim=0)[0] # [65]
            x = (x- x_min)/(x_max-x_min)
            assert torch.unique(x.min(dim=0)[0] == 0.)
            assert torch.unique(x.max(dim=0)[0] == 1.)

        if do_1d:
            n_examples = len(x)
            x = x.view(n_examples, -1)
        
        # print(torch.unique(y), torch.max(x), torch.min(x), x.shape, y.shape)

        if test_size > 0.0:
            ind_train, ind_test = train_test_split(
                np.arange(len(x)), test_size=test_size, random_state=42
            )
        else:
            ind_train = np.arange(len(x))
            ind_test = []
        x_train = x[ind_train]
        y_train = y[ind_train]
        x_test = x[ind_test]
        y_test = y[ind_test]

        # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        n_all = len(x_train)
        n_labelled_per_class = (n_all * label_proportions).astype(int)
        labelled_inds = []
        for label in y.unique():
            label_ind = np.where(y_train == label)[0]
            labelled_exs = np.random.choice(label_ind, size=n_labelled_per_class[label])
            labelled_inds.append(labelled_exs)
        labelled_inds = np.concatenate(labelled_inds)

        self.labelled_inds = labelled_inds
        x_train_labelled = x_train[labelled_inds]
        y_train_labelled = y_train[labelled_inds]

        n_all = len(x_test)
        n_labelled_per_class = (n_all * label_proportions).astype(int)
        labelled_inds = []
        for label in y.unique():
            label_ind = np.where(y_test == label)[0]
            labelled_exs = np.random.choice(label_ind, size=n_labelled_per_class[label])
            labelled_inds.append(labelled_exs)
        labelled_inds = np.concatenate(labelled_inds)

        self.labelled_inds = labelled_inds
        x_test_labelled = x_test[labelled_inds]
        y_test_labelled = y_test[labelled_inds]

        assert not (np.isin(np.unique(y_train_labelled), non_labelled)).any()
        self.train_dataset = TensorDataset(x_train, y_train) # 0 to 5
        self.train_dataset_labelled = TensorDataset(x_train_labelled, y_train_labelled) # 0 to 5
        self.test_dataset = TensorDataset(x_test, y_test) # 0 to 5
        self.test_dataset_labelled = TensorDataset(x_test_labelled, y_test_labelled) # 0 to 5
        self.full_dataset = TensorDataset(x_all, y_all)

# LABELLED_PROPORTIONS = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
# LABELLED_PROPORTIONS = LABELLED_PROPORTIONS / LABELLED_PROPORTIONS.sum()

# LABELLED_FRACTION = 0.5

# DATASET = TrentoDataset(
#     labelled_proportions=LABELLED_PROPORTIONS,
#     labelled_fraction=LABELLED_FRACTION
# )
# x,y = DATASET.train_dataset.tensors # 12085
# print(x.shape, y.shape, torch.unique(y))

# x,y = DATASET.train_dataset_labelled.tensors # 6489 subset train  
# print(x.shape, y.shape, torch.unique(y))

# x,y = DATASET.test_dataset.tensors # 15107
# print(x.shape, y.shape, torch.unique(y))

# x,y = DATASET.full_dataset.tensors # 15107
# print(x.shape, y.shape, torch.unique(y))