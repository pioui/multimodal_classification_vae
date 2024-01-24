"""
This Python script serves as a configuration file specifically designed for the Houston dataset, 
which comprises two seperate patched modalities as input. The configuration file plays a crucial role in 
defining and organizing the settings, parameters, and options required to effectively process 
and analyze the Houston dataset.
"""

import numpy as np
import logging
import os
from mcvae.dataset import houston_dataset
import torch.nn as nn
from mcvae.architectures.encoders import (
    encoder_B5,
    bernoulli_decoder_A5,
    encoder_B6,
    bernoulli_decoder_A6,
    encoder_B8,
    bernoulli_decoder_A8
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
    '#000000',
    '#3cb44b', 
    '#aaffc3', 
    '#bfef45', 
    '#f58231',
    '#ffd8b1', 
    '#9A6324', 
    '#469990', 
    '#911eb4', 
    '#dcbeff', 
    '#000075', 
    '#a9a9a9', 
    '#ffffff', 
    '#4363d8', 
    '#42d4f4', 
    '#ffe119', 
    '#800000', 
    '#e6194B', 
    '#fabed4', 
    '#f032e6', 
    '#fffac8', 
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
SAMPLES_PER_CLASS = 500

logging.basicConfig(filename = f'{outputs_dir}{PROJECT_NAME}_logs.log')

SCENARIOS = [  
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_latent=True,
        counts=None,
        n_latent = 5,
        model_name="M1M2_encoder_B0_L05",
        encoder_z1=nn.ModuleDict(
            {
                "default": encoder_B5( 
                n_input=N_INPUT,
                n_output=5,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        x_decoder=bernoulli_decoder_A5( 
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
        model_name="M1M2_encoder_B6_L30",
        encoder_z1=nn.ModuleDict(
            {
                "default": encoder_B6( 
                n_input=N_INPUT,
                n_output=30,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        x_decoder=bernoulli_decoder_A6( 
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
        model_name="M1M2_encoder_B6_L30",
        encoder_z1=nn.ModuleDict(
            {
                "default": encoder_B8( 
                n_input=N_INPUT,
                n_output=30,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        ),
        x_decoder=bernoulli_decoder_A8( 
                n_input=30,
                n_output=N_INPUT,
                dropout_rate=0,
                do_batch_norm=False,
        ),
    ),

]