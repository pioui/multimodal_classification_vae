import numpy as np
import logging
import os
from mcvae.dataset import HoustonDataset
import torch.nn as nn
from mcvae.architectures.trento_encoders import (
    EncoderB0,
    EncoderB1,
    EncoderB2,
    EncoderB3,
    EncoderB4,
)

data_dir = "/home/plo026/data/Houston/"
outputs_dir = "outputs/"
labels = [
    "Unknown", "Healthy Grass", "Stressed Grass", "Artificial Turf", "Evergreen Trees", 
    "Deciduous Trees", "Bare Earth", "Water", "Residential buildings",
    "Non-residential buildings", "Roads", "Sidewalks", "Crosswalks",
    "Major thoroughfaces", "Highways", "Railways", "Paved parking lots", "Unpaved parking lots",
    "Cars", "Trains", "Stadium seats"
    ]
color = [
    "black", "chartreuse1", "chartreuse2", "chartreuse3", "chartreuse4", 
    "darkgreen", "chocolate4", "aqua", "white", 
    "lightpink3",  "red", "darkgray", "dimgray",
    "firebrick4", "violetred4", "darkorange3", "yellow", "orange",
    "magenta", "blue", "skyblue3"
    ]
images_dir =  "images/"

N_EPOCHS = 10
LR = 1e-3
N_PARTICULES = 30
N_LATENT = 10
N_HIDDEN = 128
N_EXPERIMENTS = 1
NUM = 300
N_INPUT = 57
N_LABELS = 20
CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 25
BATCH_SIZE = 512
PROJECT_NAME = "houston"

DATASET = HoustonDataset(
    data_dir = data_dir,
)

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

logging.basicConfig(filename = f'{outputs_dir}houston_logs.log')

SCENARIOS = [  # WAKE updates
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        model_name="EncoderB0_L05_VAE",
        n_latent = 5,
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB0( 
                n_input=N_INPUT,
                n_output=5,
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
        model_name="EncoderB0_L10_VAE",
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
        n_latent = 15,
        model_name="EncoderB0_L15_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB0( 
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
        n_latent = 20,
        model_name="EncoderB0_L20_VAE",
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
        n_latent =5,
        model_name="EncoderB1_L05_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB1( 
                n_input=N_INPUT,
                n_output=5,
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
        model_name="EncoderB1_L10_VAE",
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
        n_latent=15,
        model_name="EncoderB1_L15_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB1( 
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
        model_name="EncoderB1_L20_VAE",
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
        n_latent=5,
        model_name="EncoderB2_L05_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB2( 
                n_input=N_INPUT,
                n_output=5,
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
        model_name="EncoderB2_L10_VAE",
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
        n_latent=15,
        model_name="EncoderB2_L15_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB2( 
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
        model_name="EncoderB2_L20_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB2( 
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
        n_latent=5,
        model_name="EncoderB3_L05_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB3( 
                n_input=N_INPUT,
                n_output=5,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size = 7,
            )}
        ),
    ),
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=10,
        model_name="EncoderB3_L10_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB3( 
                n_input=N_INPUT,
                n_output=10,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size = 7,
            )}
        ),
    ),
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=15,
        model_name="EncoderB3_L15_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB3( 
                n_input=N_INPUT,
                n_output=15,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size = 7,
            )}
        ),
    ),
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=20,
        model_name="EncoderB3_L20_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB3( 
                n_input=N_INPUT,
                n_output=20,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size = 7,
            )}
        ),
    ),


    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=5,
        model_name="EncoderB4_L05_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB4( 
                n_input=N_INPUT,
                n_output=5,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size = 7,
            )}
        ),
    ),
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=10,
        model_name="EncoderB4_L10_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB4( 
                n_input=N_INPUT,
                n_output=10,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size = 7,
            )}
        ),
    ),
        dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=15,
        model_name="EncoderB4_L15_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB4( 
                n_input=N_INPUT,
                n_output=15,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size = 7,
            )}
        ),
    ),
        dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent=20,
        model_name="EncoderB4_L20_VAE",
        encoder_z1=nn.ModuleDict(
            {"default": EncoderB4( 
                n_input=N_INPUT,
                n_output=20,
                n_hidden=512,
                dropout_rate=0,
                do_batch_norm=False,
                kernel_size = 7,
            )}
        ),
    ),
    
]