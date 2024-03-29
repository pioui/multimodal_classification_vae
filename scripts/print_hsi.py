"""
This Python script creates the false color representation from the HSI information for Houston and Trento dataset

Usage:
  python3 scripts/print_hsi.py 

"""

from email.mime import image
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from scipy import io
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score
import os
import numpy as np
import torch
import tifffile
import csv
import pylab as pl
import pandas as pd
import seaborn as sns
from PIL import Image

from mcvae.utils import generate_latex_matrix_from_dict, generate_latex_confusion_matrix, crop_npy

print(os.listdir('outputs/'))

for project_name in os.listdir('outputs/'):
    if project_name == 'trento':
        dataset = 'trento'
        from trento_config import *
    
    elif project_name == 'houston':
        dataset = 'houston'
        from houston_config import *
    
    else:
        continue

    if dataset == "trento":
        from mcvae.dataset import trento_dataset
        DATASET = trento_dataset(data_dir=data_dir, 
        )
        (r,g,b) = (31,17,8)
        li = 2
        hi = 63
    elif dataset == "houston":
        from mcvae.dataset import houston_dataset
        DATASET = houston_dataset(data_dir=data_dir, samples_per_class=SAMPLES_PER_CLASS, 
        )
        li = 7
        hi = 50
        (r,g,b) = (16,13,6) #real


    X_train,y_train = DATASET.train_dataset_labelled.tensors 
    X_test,y_test = DATASET.test_dataset_labelled.tensors 
    X,y = DATASET.full_dataset.tensors 

    X_r = X[:,r].reshape(SHAPE)
    X_g = X[:,g].reshape(SHAPE)
    X_b = X[:,b].reshape(SHAPE)
    X_rgb = np.stack((X_r, X_g, X_b), axis = 2)
    plt.imshow(X_rgb, interpolation='nearest', 
    vmin=0, vmax=1, 
    # cmap='gray'
    )
    plt.axis('off')
    plt.savefig(f"{images_dir}{dataset}_RGB_HSI.png",bbox_inches='tight', pad_inches=0, dpi=500)

