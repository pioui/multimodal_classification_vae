from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torch

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
    ) -> None:
        """
            Definition of trento Dataset loader
            The data are loaded from data_dir
            only labelled data are used (non zero labels)
            the lidar and hyperspectral data are concatenated 
            so for each pixel we have a feature array with 65 elements, 
            the 2 first are lidar and therest hypersoectral

            label_fraction: the precentage of the training data to use labelled
            validation_fraction: the precentage of the test set that is gonna be used as validation

            we keep 90% of the train set labelled and the rest 10% as unlabelled
        """
        super().__init__()

        

        image_hyper = pixelwise_reshape(torch.tensor(tifffile.imread(data_dir+"hyper_Italy.tif"))) # [99600,63]
        image_lidar = pixelwise_reshape(torch.tensor(tifffile.imread(data_dir+"LiDAR_Italy.tif"))) # [99600,2]
        x = torch.cat((image_hyper,image_lidar), dim = 1) # 99600,65

        # Normalization 
        #TODO: this can be written (more tidy) as a transform with other preprocessin' when making the dataset 
        mean = torch.mean(x, dim = 0)
        std= torch.std(x, dim = 0)
        x = (x - mean)/std # [99600,65]

        y = pixelwise_reshape(
            torch.tensor(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"], dtype = torch.int64)
        ) # [99600] 0 to 6

        valid_indeces = (y!=0).nonzero()[:,0] 
        X = x[valid_indeces] # [30214, 65]
        Y = y[valid_indeces] # [30214] 1 to 6

        # sample if needed
        RDM_INDICES = np.random.choice(len(X), 5000) 
        X = X[RDM_INDICES]
        Y = Y[RDM_INDICES]

        # 50-50 train and test
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            X, Y, train_size = 0.5, stratify = Y
        ) # [15107,65] 1 to 6

        # 80-20 train and validation
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, train_size = 0.8, stratify = y_train_val
        ) # [12085, 65], [3022, 65] 1 to 6 

        # 60,40 train labelled and unlabelled
        excluded_label = torch.max(torch.unique(y_train))
        labelled_indeces = (y_train!=excluded_label).nonzero()[:,0]
        

        self.labelled_fraction = 0.6

        x_train_labelled, _, y_train_labelled, _ = train_test_split(
            x_train[labelled_indeces], y_train[labelled_indeces], train_size = 0.6, stratify = y_train[labelled_indeces]
        )  # [6489,65] 1 to 5 

        # print("x:", x.shape, y.shape, torch.unique(y))
        # print("X:", X.shape, Y.shape, torch.unique(Y))
        # print("x_train_val: ",x_train_val.shape, y_train_val.shape, torch.unique(y_train_val)) 
        # print("x_train: ",x_train.shape, y_train.shape, torch.unique(y_train)) 
        # print("x_train_labelled: ", x_train_labelled.shape, y_train_labelled.shape, torch.unique(y_train_labelled)) 
        # print("x_val: ",x_val.shape, y_val.shape, torch.unique(y_val))
        # print("x_test: ", x_test.shape, y_test.shape, torch.unique(y_test)) 

        # NOTE: 1 is substracted from the labels so the smaller label is 0, makes things simpler for the way it's implimented now.
        self.train_dataset = TensorDataset(x_train, y_train-1) 
        self.train_dataset_labelled = TensorDataset(x_train_labelled, y_train_labelled-1) 
        self.validation_dataset = TensorDataset(x_val, y_val-1)
        self.test_dataset = TensorDataset(x_test, y_test-1)

        # self.train_dataset = TensorDataset(x_train[:200,:], y_train[:200]-1)  #819
        # self.train_dataset_labelled = TensorDataset(x_train_labelled[:20,:], y_train_labelled[:20]-1)  #34
        # self.test_dataset = TensorDataset(x_test[:300,:], y_test[:300]-1) #29395



DATASET = TrentoDataset()

x,y = DATASET.train_dataset.tensors # 12085
print(x.shape, y.shape, torch.unique(y)) 

x,y = DATASET.train_dataset_labelled.tensors # 6489 subset train  
print(x.shape, y.shape, torch.unique(y))

x,y = DATASET.validation_dataset.tensors # 3022
print(x.shape, y.shape, torch.unique(y))

x,y = DATASET.test_dataset.tensors # 15107
print(x.shape, y.shape, torch.unique(y))


