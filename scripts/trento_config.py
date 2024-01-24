"""
This Python script serves as a configuration file specifically designed for the Trento dataset, 
which comprises two stacked modalities as input. The configuration file plays a crucial role in 
defining and organizing the settings, parameters, and options required to effectively process 
and analyze the Trento dataset.
"""

import numpy as np
import logging
import os
from mcvae.dataset import trento_dataset
import torch.nn as nn
from mcvae.architectures.encoders import (
    encoder_B1,
    encoder_B2,
    encoder_B3,
    encoder_B4
)

data_dir = "/home/pigi/data/trento/"
outputs_dir = "outputs/trento/"
images_dir =  "outputs/trento/images/"
labels = ["Unknown", "A.Trees", "Buildings", "Ground", "Wood", "Vineyards", "Roads"]
color = ["black", "red", "gray", "blue", "orange", "green","yellow"]

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

N_EPOCHS = 10
LR = 1e-4
N_PARTICULES = 30
N_HIDDEN = 128
N_EXPERIMENTS = 1
N_INPUT = 65


N_LABELS = 6
SHAPE = (166,600)
CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 25
BATCH_SIZE = 512
PROJECT_NAME = "trento"

logging.basicConfig(filename = f'{outputs_dir}{PROJECT_NAME}_logs.log')

SCENARIOS = [ 

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=10,
        model_name="M1M2_encoder_B1_L10",
        encoder_z1=nn.ModuleDict(
            {"default": encoder_B1( 
                n_input=N_INPUT,
                n_output=10,
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
        model_name="M1M2_encoder_B1_L15",
        encoder_z1=nn.ModuleDict(
            {"default": encoder_B1( 
                n_input=N_INPUT,
                n_output=15,
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
        n_latent=20,
        model_name="M1M2_encoder_B1_L20",
        encoder_z1=nn.ModuleDict(
            {"default": encoder_B1( 
                n_input=N_INPUT,
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
        n_latent=10,
        model_name="M1M2_encoder_B2_L10",
        encoder_z1=nn.ModuleDict(
            {"default": encoder_B2( 
                n_input=N_INPUT,
                n_output=10,
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
        n_latent=15,
        model_name="M1M2_encoder_B2_L15",
        encoder_z1=nn.ModuleDict(
            {"default": encoder_B2( 
                n_input=N_INPUT,
                n_output=15,
                n_hidden=256,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        batch_size=128,
    ),

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=20,
        model_name="M1M2_encoder_B2_L20",
        encoder_z1=nn.ModuleDict(
            {"default": encoder_B2( 
                n_input=N_INPUT,
                n_output=20,
                n_hidden=256,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        batch_size=128,
    ),


    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=10,
        model_name="M1M2_encoder_B3_L10",
        encoder_z1=nn.ModuleDict(
            {"default": encoder_B3( 
                n_input=N_INPUT,
                n_output=10,
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
        model_name="M1M2_encoder_B3_L15",
        encoder_z1=nn.ModuleDict(
            {"default": encoder_B3( 
                n_input=N_INPUT,
                n_output=15,
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
        n_latent=20,
        model_name="M1M2_encoder_B3_L20",
        encoder_z1=nn.ModuleDict(
            {"default": encoder_B3( 
                n_input=N_INPUT,
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
        n_latent=10,
        model_name="M1M2_encoder_B4_L10",
        encoder_z1=nn.ModuleDict(
            {"default": encoder_B4( 
                n_input=N_INPUT,
                n_output=10,
                n_hidden=512,
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
        model_name="M1M2_encoder_B4_L15",
        encoder_z1=nn.ModuleDict(
            {"default": encoder_B4( 
                n_input=N_INPUT,
                n_output=15,
                n_hidden=512,
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
        n_latent=20,
        model_name="M1M2_encoder_B4_L20",
        encoder_z1=nn.ModuleDict(
            {"default": encoder_B4( 
                n_input=N_INPUT,
                n_output=20,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
    ),
    
]