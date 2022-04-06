import numpy as np
import logging
import os
from mcvae.dataset import trentoMultimodalDataset
import torch.nn as nn
from mcvae.architectures.trento_encoders import (
    EncoderB0,
    EncoderB1,
    EncoderB2,
    EncoderB3,
    EncoderB4
)

data_dir = "/home/plo026/data/trento_multimodal/"
outputs_dir = "outputs/trento_multimodal/"
labels = ["Unknown", "A.Trees", "Buildings", "Ground", "Wood", "Vineyards", "Roads"]
color = ["black", "red", "gray", "blue", "orange", "green","yellow"]
images_dir =  "outputs/trento_multimodal/images/"

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

N_EPOCHS = 200
LR = 1e-3
N_PARTICULES = 30
N_HIDDEN = 128
N_EXPERIMENTS = 1

N1_INPUT = 63
N2_INPUT = 2
N_LABELS = 6
SHAPE = (166,600)
CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 25
BATCH_SIZE = 512
PROJECT_NAME = "trento_multimodal"

logging.basicConfig(filename = f'{outputs_dir}{PROJECT_NAME}_logs.log')

SCENARIOS = [  # WAKE updates

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=10,
        model_name="EncoderB1_L10_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB1( 
                n_input=N1_INPUT,
                n_output=10,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        encoder_z2=nn.ModuleDict(
            {"default": EncoderB1( 
                n_input=N2_INPUT,
                n_output=10,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
    ),
    
]