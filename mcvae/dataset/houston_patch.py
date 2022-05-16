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

class houstonPatchDataset(Dataset):
    def __init__(
        self,
        data_dir,
        samples_per_class=200,
        train_size=0.5,
        do_preprocess=True,
        patch_size = 5 #odd number
    ) -> None:
        super().__init__()

        assert patch_size % 2 == 1
        image_hyper = torch.tensor(tifffile.imread(data_dir+"houston_hyper.tif")) # [50,1202,4768]
        image_lidar = torch.tensor(tifffile.imread(data_dir+"houston_lidar.tif")) # [7,1202,4768]
        x = torch.cat((image_hyper,image_lidar), dim = 0) # [57,1202,4768]
        x_all = x
        x_all = x_all.reshape(len(x_all),-1) # [57,5731136]
        x_all = torch.transpose(x_all, 1,0) # [5731136,57]

        #Normalize to [0,1]
        if do_preprocess: 
            x_all = normalize(x_all).float()

        x_all = torch.transpose(x_all,1,0)
        x_all = x_all.reshape(-1,1202,4768) # [57,1202,4768]

        # Patching
        x_padded = torch.nn.ReflectionPad2d(int(patch_size/2))(x_all) # [57,166+p/2, 600+p/2]
        x_patched = x_padded.unfold(dimension=1, size=patch_size, step=1)
        x_patched = x_patched.unfold(dimension=2, size=patch_size, step=1) # [65,166,600,p,p]
        x_patched = x_patched.reshape(57,-1,patch_size,patch_size) # [65,99600,p,p]
        x_all = x_patched.transpose(1,0) # [5731136,57,p,p]

        y = torch.tensor(tifffile.imread(data_dir+"houston_gt.tif"), dtype = torch.int64) # [1202,4768]
        y_all = y.reshape(-1) # [5731136] 0 to 20

        train_inds = []
        for label in y_all.unique():
            label_ind = np.where(y_all == label)[0]
            samples = samples_per_class
            if label == 0:
                labelled_exs = np.random.choice(label_ind, size=(len(y_all.unique())-1)*samples, replace=False)
            elif (len(label_ind)< samples):
                labelled_exs = np.random.choice(label_ind, size=samples, replace=True)
            else:
                labelled_exs = np.random.choice(label_ind, size=samples, replace=False)

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
        self.train_dataset = TensorDataset(x_train, y_train-1) # -1 to 19
        self.train_dataset_labelled = TensorDataset(x_train_labelled, y_train_labelled-1) # 0 to 19
        self.test_dataset = TensorDataset(x_test, y_test-1) # -1 to 19
        self.test_dataset_labelled = TensorDataset(x_test_labelled, y_test_labelled-1) # 0 to 19
        self.full_dataset = TensorDataset(x_all, y_all) # 0 to 20
        log_train_test_split([y_all, y_train, y_train_labelled, y_test, y_test_labelled])

if __name__ == "__main__":

    DATASET = houstonPatchDataset(
        data_dir = "/home/plo026/data/houston/",
    )
    x,y = DATASET.train_dataset.tensors # 819
    print(x.shape, y.shape, torch.unique(y))
    plt.imshow(x[1000,9])
    plt.show()
    print(y[1000])

    # x,y = DATASET.train_dataset_labelled.tensors # 409 
    # print(x.shape, y.shape, torch.unique(y))

    # x,y = DATASET.test_dataset.tensors # 15107
    # print(x.shape, y.shape, torch.unique(y))

    # x,y = DATASET.test_dataset_labelled.tensors # 15107
    # print(x.shape, y.shape, torch.unique(y))

    # x,y = DATASET.full_dataset.tensors # 15107
    # print(x.shape, y.shape, torch.unique(y))