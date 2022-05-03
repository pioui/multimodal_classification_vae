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

data_dir = "/Users/plo026/data/houston/"
outputs_dir = "outputs/houston_multimodal/"
labels = [
    "Unknown", "Healthy Grass", "Stressed Grass", "Artificial Turf", "Evergreen Trees", 
    "Deciduous Trees", "Bare Earth", "Water", "Residential buildings",
    "Non-residential buildings", "Roads", "Sidewalks", "Crosswalks",
    "Major thoroughfaces", "Highways", "Railways", "Paved parking lots", "Unpaved parking lots",
    "Cars", "Trains", "Stadium seats"
    ]
color = [
    "black", "limegreen", "lime", "forestgreen", "green", 
    "darkgreen", "saddlebrown", "aqua", "white", 
    "plum",  "red", "darkgray", "dimgray",
    "firebrick", "darkred", "peru", "yellow", "orange",
    "magenta", "blue", "skyblue"
    ]
images_dir =  "outputs/houston_multimodal/images/"

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

N_EPOCHS = 1000
LR = 1e-4
N_PARTICULES = 30
N_HIDDEN = 128
N_EXPERIMENTS = 1
N1_INPUT = 50
N2_INPUT = 7
N_LABELS = 20
SHAPE = (1202,4768)
CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 25
BATCH_SIZE = 512
PROJECT_NAME = "houston_multimodal"
SAMPLES_PER_CLASS = 2000

logging.basicConfig(filename = f'{outputs_dir}{PROJECT_NAME}_logs.log')


SCENARIOS = [  # WAKE updates

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=30,
        model_name="EncoderB0_L30_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB0( 
                n_input=N1_INPUT,
                n_output=30,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        encoder_z2=nn.ModuleDict(
            {"default": EncoderB0( 
                n_input=N2_INPUT,
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
        n_latent=30,
        model_name="EncoderB1_L30_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB1( 
                n_input=N1_INPUT,
                n_output=30,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        encoder_z2=nn.ModuleDict(
            {"default": EncoderB1( 
                n_input=N2_INPUT,
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
        n_latent=30,
        model_name="EncoderB2_L30_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB2( 
                n_input=N1_INPUT,
                n_output=30,
                n_hidden=256,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        encoder_z2=nn.ModuleDict(
            {"default": EncoderB2( 
                n_input=N2_INPUT,
                n_output=30,
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
        model_name="EncoderB3_L30_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB3( 
                n_input=N1_INPUT,
                n_output=30,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        encoder_z2=nn.ModuleDict(
            {"default": EncoderB3( 
                n_input=N2_INPUT,
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
        n_latent=30,
        model_name="EncoderB4_L30_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB4( 
                n_input=N1_INPUT,
                n_output=30,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        encoder_z2=nn.ModuleDict(
            {"default": EncoderB4( 
                n_input=N2_INPUT,
                n_output=30,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
    ),
    
]