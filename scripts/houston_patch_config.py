"""
Configuation file for houston dataset with two stacked patched input

"""

import numpy as np
import logging
import os
from mcvae.dataset import houstonDataset
import torch.nn as nn
from mcvae.architectures.encoders import (
    EncoderB5,
    BernoulliDecoderA5,
    EncoderB6,
    BernoulliDecoderA6,
    EncoderB8,
    BernoulliDecoderA8
)

data_dir = "/home/pigi/data/houston/"
outputs_dir = "outputs/houston_patch/"
images_dir =  "outputs/houston_patch/images/"

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
PATCH_SIZE = 3
N_LABELS = 20
SHAPE = (1202,4768)
CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 25
BATCH_SIZE = 128
PROJECT_NAME = "houston_patch"
SAMPLES_PER_CLASS = 200

logging.basicConfig(filename = f'{outputs_dir}{PROJECT_NAME}_logs.log')

SCENARIOS = [  
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent = 5,
        model_name="M1M2_EncoderB0_L05",
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

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent = 30,
        model_name="M1M2_EncoderB6_L30",
        encoder_z1=nn.ModuleDict(
            {
                "default": EncoderB6( 
                n_input=N_INPUT,
                n_output=30,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        x_decoder=BernoulliDecoderA6( 
                n_input=30,
                n_output=N_INPUT,
                dropout_rate=0,
                do_batch_norm=False,
        ),
    ),

    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent = 30,
        model_name="M1M2_EncoderB6_L30",
        encoder_z1=nn.ModuleDict(
            {
                "default": EncoderB8( 
                n_input=N_INPUT,
                n_output=30,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        x_decoder=BernoulliDecoderA8( 
                n_input=30,
                n_output=N_INPUT,
                dropout_rate=0,
                do_batch_norm=False,
        ),
    ),

]