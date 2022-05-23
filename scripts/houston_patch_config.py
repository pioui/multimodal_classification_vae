import numpy as np
import logging
import os
from mcvae.dataset import trentoDataset
import torch.nn as nn
from mcvae.architectures.encoders import (
    EncoderB5,
    BernoulliDecoderA5,
    EncoderB6,
    BernoulliDecoderA6,
    EncoderB8,
    BernoulliDecoderA8
)

data_dir = "/home/plo026/data/houston/"
outputs_dir = "outputs/houston_patch/"
labels = [
    "Unknown", "Healthy Grass", "Stressed Grass", "Artificial Turf", "Evergreen Trees", 
    "Deciduous Trees", "Bare Earth", "Water", "Residential buildings",
    "Non-residential buildings", "Roads", "Sidewalks", "Crosswalks",
    "Major thoroughfares", "Highways", "Railways", "Paved parking lots", "Unpaved parking lots",
    "Cars", "Trains", "Stadium seats"
    ]
color = [
    "black", "limegreen", "lime", "forestgreen", "green", 
    "darkgreen", "saddlebrown", "aqua", "white", 
    "plum",  "red", "darkgray", "dimgray",
    "firebrick", "darkred", "peru", "yellow", "orange",
    "magenta", "blue", "skyblue"
    ]
images_dir =  "outputs/houston_patch/images/"
heterophil_matrix = np.array(
    [
        [1,2,3,5,5,4,6,6,6,5,5,5,5,5,6,5,5,6,6,6],
        [2,1,3,5,5,4,6,6,6,5,5,5,5,5,6,5,5,6,6,6],
        [3,3,1,5,5,4,6,6,6,5,5,5,5,5,6,5,5,6,6,6],
        [5,5,5,1,2,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],
        [5,5,5,2,1,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],
        [4,4,4,6,6,1,6,6,6,5,5,5,5,5,6,5,2,6,6,6],
        [6,6,6,6,6,6,1,6,6,6,6,6,6,6,6,6,6,6,6,6],
        [6,6,6,6,6,6,6,1,2,5,5,5,5,5,6,5,6,6,6,6],
        [6,6,6,6,6,6,6,2,1,5,5,5,5,5,6,5,6,6,6,6],
        [5,5,5,6,6,5,6,5,5,1,2,2,2,2,6,2,6,6,6,6],
        [5,5,5,6,6,5,6,5,5,2,1,3,2,2,6,2,6,6,6,6],
        [5,5,5,6,6,5,6,5,5,3,3,1,3,3,6,4,6,6,6,6],
        [5,5,5,6,6,5,6,5,5,2,2,3,1,2,6,2,6,6,6,6],
        [5,5,5,6,6,5,6,5,5,2,2,3,2,1,6,2,6,6,6,6],
        [6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,6,6,3,3,6],
        [5,5,5,6,6,5,6,5,5,2,2,4,2,2,6,1,3,6,6,6],
        [5,5,5,6,6,2,6,6,6,6,6,6,6,6,6,3,1,6,6,6],
        [6,6,6,6,6,6,6,6,6,6,6,6,6,6,3,6,6,1,3,6],
        [6,6,6,6,6,6,6,6,6,6,6,6,6,6,3,6,6,3,1,6],
        [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1],
    ]
)


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

SCENARIOS = [  # WAKE updates
    # dict(
    #     loss_gen="ELBO",
    #     loss_wvar="ELBO",
    #     reparam_latent=True,
    #     counts=None,
    #     n_latent = 5,
    #     model_name="EncoderB0_L05_VAE",
    #     encoder_z1=nn.ModuleDict(
    #         {
    #             "default": EncoderB5( 
    #             n_input=N_INPUT,
    #             n_output=5,
    #             n_hidden=128,
    #             dropout_rate=0,
    #             do_batch_norm=False,
    #         )}
    #     ),
    #     x_decoder=BernoulliDecoderA5( 
    #             n_input=5,
    #             n_output=N_INPUT,
    #             dropout_rate=0,
    #             do_batch_norm=False,
    #     ),
    # ),
    #     dict(
    #     loss_gen="ELBO",
    #     loss_wvar="ELBO",
    #     reparam_latent=True,
    #     counts=None,
    #     n_latent = 30,
    #     model_name="EncoderB6_L30_VAE",
    #     encoder_z1=nn.ModuleDict(
    #         {
    #             "default": EncoderB6( 
    #             n_input=N_INPUT,
    #             n_output=30,
    #             n_hidden=128,
    #             dropout_rate=0,
    #             do_batch_norm=False,
    #         )}
    #     ),
    #     x_decoder=BernoulliDecoderA6( 
    #             n_input=30,
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
        n_latent = 30,
        model_name="EncoderB6_L3o_VAE",
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