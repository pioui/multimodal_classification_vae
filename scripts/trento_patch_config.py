import numpy as np
import logging
import os
from mcvae.dataset import trentoDataset
import torch.nn as nn
from mcvae.architectures.encoders import (
    EncoderB0,
    EncoderB1,
    EncoderB2,
    EncoderB3,
    EncoderB4,
    EncoderB5,
    BernoulliDecoderA5
)

data_dir = "/Users/plo026/data/trento/"
outputs_dir = "outputs/trento/"
labels = ["Unknown", "A.Trees", "Buildings", "Ground", "Wood", "Vineyards", "Roads"]
color = ["black", "red", "gray", "blue", "orange", "green","yellow"]
images_dir =  "outputs/trento/images/"

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

N_EPOCHS = 1
LR = 1e-4
N_PARTICULES = 30
N_HIDDEN = 128
N_EXPERIMENTS = 1
N_INPUT = 65
N_PATCH = 13
N_LABELS = 6
SHAPE = (166,600)
CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 25
BATCH_SIZE = 512
PROJECT_NAME = "trento"

logging.basicConfig(filename = f'{outputs_dir}{PROJECT_NAME}_logs.log')

SCENARIOS = [  # WAKE updates
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent = 5,
        model_name="EncoderB0_L05_VAE",
        encoder_z1=nn.ModuleDict(
            {
                "default": EncoderB5( 
                n_input=N_INPUT,
                n_output=5,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        x_decoder=BernoulliDecoderA5( 
                n_input=5,
                n_output=N_INPUT,
                dropout_rate=0,
                do_batch_norm=False,
        ),
    ),
]