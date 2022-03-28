import numpy as np
import logging
import os

data_dir = "/Users/plo026/data/Trento/"
outputs_dir = "outputs/"
labels = ["Unknown", "Apple Trees", "Buildings", "Ground", "Wood", "Vineyard", "Roads"]
color = ["black", "red", "gray", "blue", "orange", "green","yellow"]

N_EPOCHS = 10
LR = 1e-3
N_PARTICULES = 30
N_LATENT = 10
N_HIDDEN = 128
N_EXPERIMENTS = 1
NUM = 300
LABELLED_PROPORTIONS = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
LABELLED_PROPORTIONS = LABELLED_PROPORTIONS / LABELLED_PROPORTIONS.sum()
LABELLED_FRACTION = 0.5 
N_INPUT = 65
N_LABELS = 6
CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 25
BATCH_SIZE = 512
TEST_SIZE = 0.5
TOTAL_SIZE = 0.1

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
logging.basicConfig(filename = f'{outputs_dir}trento_logs.log',level=logging.DEBUG)
