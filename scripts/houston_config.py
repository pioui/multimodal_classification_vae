"""
This Python script serves as a configuration file specifically designed for the Houston dataset, 
which comprises two stacked modalities as input. The configuration file plays a crucial role in 
defining and organizing the settings, parameters, and options required to effectively process 
and analyze the Houston dataset.
"""

import numpy as np
import logging
import os
from mcvae.dataset import houstonDataset
import torch.nn as nn
from mcvae.architectures.encoders import (
    EncoderB0,
    EncoderB1,
    EncoderB2,
    EncoderB3,
    EncoderB4,
)

data_dir = "/home/pigi/data/houston/"
outputs_dir = "outputs/houston/"
images_dir =  "outputs/houston/images/"

labels = [
    "Unknown", 
    "Healthy Grass", 
    "Stressed Grass", 
    "Artificial Turf", 
    "Evergreen Trees", 
    "Deciduous Trees", 
    "Bare Earth", 
    "Water", 
    "Residential buildings",
    "Non-residential buildings", 
    "Roads", 
    "Sidewalks", 
    "Crosswalks",
    "Major thoroughfares", 
    "Highways", 
    "Railways", 
    "Paved parking lots", 
    "Unpaved parking lots",
    "Cars", 
    "Trains", 
    "Stadium seats"
    ]
color = [
    "black", 
    "limegreen", 
    "lime", 
    "forestgreen", 
    "green", 
    "darkgreen", 
    "saddlebrown", 
    "aqua", 
    "white", 
    "plum",  
    "red", 
    "darkgray", 
    "dimgray",
    "firebrick", 
    "darkred", 
    "peru", 
    "yellow", 
    "orange",
    "magenta", 
    "blue", 
    "skyblue"
    ]

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

N_EPOCHS = 1000
LR = 1e-4
N_PARTICULES = 30
N_HIDDEN = 128
N_EXPERIMENTS = 1
N_INPUT = 57

N_LABELS = 20
SHAPE = (1202,4768)
CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 25
BATCH_SIZE = 1024
PROJECT_NAME = "houston"
SAMPLES_PER_CLASS = 2000

logging.basicConfig(filename = f'{outputs_dir}{PROJECT_NAME}_logs.log')

SCENARIOS = [  # WAKE updates
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        model_name="M1M2_EncoderB0_L10",
        n_latent = 10,
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB0( 
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
        model_name="M1M2_EncoderB0_L20",
        n_latent = 20,
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB0( 
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
        n_latent = 30,
        model_name="M1M2_EncoderB0_L30",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB0( 
                n_input=N_INPUT,
                n_output=30,
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
        n_latent = 40,
        model_name="M1M2_EncoderB0_L40",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB0( 
                n_input=N_INPUT,
                n_output=40,
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
        n_latent =10,
        model_name="M1M2_EncoderB1_L10",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB1( 
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
        n_latent=20,
        model_name="M1M2_EncoderB1_L20",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB1( 
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
        n_latent=30,
        model_name="M1M2_EncoderB1_L30",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB1( 
                n_input=N_INPUT,
                n_output=30,
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
        n_latent=40,
        model_name="M1M2_EncoderB1_L40",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB1( 
                n_input=N_INPUT,
                n_output=40,
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
        model_name="M1M2_EncoderB2_L10",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB2( 
                n_input=N_INPUT,
                n_output=10,
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
        model_name="M1M2_EncoderB2_L20",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB2( 
                n_input=N_INPUT,
                n_output=20,
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
        n_latent=30,
        model_name="M1M2_EncoderB2_L30",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB2( 
                n_input=N_INPUT,
                n_output=30,
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
        n_latent=40,
        model_name="M1M2_EncoderB2_L40",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB2( 
                n_input=N_INPUT,
                n_output=40,
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
        model_name="M1M2_EncoderB3_L10",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB3( 
                n_input=N_INPUT,
                n_output=10,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size=9
            )}
        ),
    ),

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=20,
        model_name="M1M2_EncoderB3_L20",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB3( 
                n_input=N_INPUT,
                n_output=20,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size=9
            )}
        ),
    ),

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=30,
        model_name="M1M2_EncoderB3_L30",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB3( 
                n_input=N_INPUT,
                n_output=30,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size=9
            )}
        ),
    ),

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=40,
        model_name="M1M2_EncoderB3_L40",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB3( 
                n_input=N_INPUT,
                n_output=40,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size=9
            )}
        ),
    ),

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=10,
        model_name="M1M2_EncoderB4_L10",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB4( 
                n_input=N_INPUT,
                n_output=10,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size=9
            )}
        ),
    ),
    
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=20,
        model_name="M1M2_EncoderB4_L20",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB4( 
                n_input=N_INPUT,
                n_output=20,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size=9
            )}
        ),
    ),
    
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=30,
        model_name="M1M2_EncoderB4_L30",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB4( 
                n_input=N_INPUT,
                n_output=30,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size=9
            )}
        ),
    ),

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=40,
        model_name="M1M2_EncoderB4_L40",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB4( 
                n_input=N_INPUT,
                n_output=40,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size=9
            )}
        ),
    ),
    
]