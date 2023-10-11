"""
This Python script serves as a configuration file specifically designed for the Houston dataset, 
which comprises two stacked patched modalities as input. The configuration file plays a crucial role in 
defining and organizing the settings, parameters, and options required to effectively process 
and analyze the Houston dataset.
"""

import numpy as np
import logging
import os
from mcvae.dataset import trento_dataset
import torch.nn as nn
from mcvae.architectures.encoders import (
    encoder_B5,
    bernoulli_decoder_A5,
    encoder_B6,
    bernoulli_decoder_A6,
    encoder_B8,
    bernoulli_decoder_A8
)

data_dir = "/home/pigi/data/trento/"
outputs_dir = "outputs/trento_patch/"
labels = ["Unknown", "A.Trees", "Buildings", "Ground", "Wood", "Vineyards", "Roads"]
color = ["black", "red", "gray", "blue", "orange", "green","yellow"]
images_dir =  "outputs/trento_patch/images/"

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

PATCH_SIZE = 5
N_LABELS = 6
SHAPE = (166,600)
CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 25
BATCH_SIZE = 64
PROJECT_NAME = "trento_patch"

logging.basicConfig(filename = f'{outputs_dir}{PROJECT_NAME}_logs.log')

SCENARIOS = [  
    # dict(
    #     loss_gen="ELBO",
    #     loss_wvar="ELBO",
    #     reparam_latent=True,
    #     counts=None,
    #     n_latent = 5,
    #     model_name="M1M2_encoder_B0_L05",
    #     encoder_z1=nn.ModuleDict(
    #         {
    #             "default": encoder_B5( 
    #             n_input=N_INPUT,
    #             n_output=5,
    #             n_hidden=128,
    #             dropout_rate=0,
    #             do_batch_norm=False,
    #         )}
    #     ),
    #     x_decoder=bernoulli_decoder_A5( 
    #             n_input=5,
    #             n_output=N_INPUT,
    #             dropout_rate=0,
    #             do_batch_norm=False,
    #     ),
    # ),
    
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent = 15,
        model_name="M1M2_encoder_B6_L15",
        encoder_z1=nn.ModuleDict(
            {
                "default": encoder_B6( 
                n_input=N_INPUT,
                n_output=15,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        x_decoder=bernoulli_decoder_A6( 
                n_input=15,
                n_output=N_INPUT,
                dropout_rate=0,
                do_batch_norm=False,
        ),
    ),

    # dict(
    #     loss_gen="ELBO",
    #     loss_wvar="ELBO",
    #     reparam_latent=True,
    #     counts=None,
    #     n_latent = 15,
    #     model_name="M1M2_encoder_B6_L15",
    #     encoder_z1=nn.ModuleDict(
    #         {
    #             "default": encoder_B8( 
    #             n_input=N_INPUT,
    #             n_output=15,
    #             n_hidden=128,
    #             dropout_rate=0,
    #             do_batch_norm=False,
    #         )}
    #     ),
    #     x_decoder=bernoulli_decoder_A8( 
    #             n_input=15,
    #             n_output=N_INPUT,
    #             dropout_rate=0,
    #             do_batch_norm=False,
    #     ),
    # ),
    
]