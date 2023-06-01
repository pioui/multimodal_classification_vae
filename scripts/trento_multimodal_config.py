"""
This Python script serves as a configuration file specifically designed for the Trento dataset, 
which comprises two seperate modalities as input. The configuration file plays a crucial role in 
defining and organizing the settings, parameters, and options required to effectively process 
and analyze the Trento dataset.
"""

import numpy as np
import logging
import os
from mcvae.dataset import trentoMultimodalDataset
import torch.nn as nn
from mcvae.architectures.encoders import (
    EncoderB0,
    EncoderB2,
    EncoderB4
)

data_dir = "/home/pigi/data/trento/"
outputs_dir = "outputs/trento_multimodal/"
images_dir =  "outputs/trento_multimodal/images/"
labels = ["Unknown", "A.Trees", "Buildings", "Ground", "Wood", "Vineyards", "Roads"]
color = ["black", "red", "gray", "blue", "orange", "green","yellow"]

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

N_EPOCHS = 1000
LR = 1e-4
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

SCENARIOS = [ 
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=20,
        model_name="multi-M1M2_EncoderB0_L20",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB0( 
                n_input=N1_INPUT,
                n_output=20,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        encoder_z2=nn.ModuleDict(
            {"default": EncoderB0( 
                n_input=N2_INPUT,
                n_output=20,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
    ),

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=15,
        model_name="multi-M1M2_EncoderB2_L15",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB2( 
                n_input=N1_INPUT,
                n_output=15,
                n_hidden=256,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        encoder_z2=nn.ModuleDict(
            {"default": EncoderB2( 
                n_input=N2_INPUT,
                n_output=15,
                n_hidden=256,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
    ),

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=10,
        model_name="multi-M1M2_EncoderB4_L10",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB4( 
                n_input=N1_INPUT,
                n_output=10,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        encoder_z2=nn.ModuleDict(
            {"default": EncoderB4( 
                n_input=N2_INPUT,
                n_output=10,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
    ),
    
]