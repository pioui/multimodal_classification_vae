from mcvae.models.regular_modules import ClassifierA, EncoderB
from mcvae.models.semi_supervised_vae_relaxed import RelaxedSVAE
import tifffile
import numpy as np
import torch
import os
from torch import nn
from scipy import io

from sklearn.metrics import classification_report, confusion_matrix
from trento_utils import compute_reject_score
from mcvae.dataset import TrentoDataset

from mcvae.models.trento_encoders import (
    EncoderB3,
)
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torch.distributions import (
    RelaxedOneHotCategorical,
)

def pixelwise_reshape(input):
    """
    compress
    """
    if len(input.shape)==3:
        return input.reshape(-1,input.shape[0])
    if len(input.shape)==2:
        return input.reshape(-1)

PATH_C =  "/Users/plo026/repos/decision-making-vaes/models/trento-relaxed_nparticules_30/classifier_-8050918908592526525.pt"
PATH_E =  "/Users/plo026/repos/decision-making-vaes/models/trento-relaxed_nparticules_30/encoder_z1_-8050918908592526525.pt"

N_INPUT = 65
N_LABELS = 5 #one label is excluded
N_LATENT = 10
N_HIDDEN = 128
N_SAMPLES = 20

multi_encoder_keys: list = ["default"]

encoder =nn.ModuleDict(
            {"default": EncoderB3( 
                n_input=N_INPUT,
                n_output=N_LATENT,
                n_hidden=128,
                dropout_rate=0,
                do_batch_norm=False,
            )}
        )
if os.path.exists(PATH_E):
    print("model exists; loadizng from .pt")
    encoder.load_state_dict(torch.load(PATH_E))
encoder.eval()

classifier = nn.ModuleDict(
            {
                "default": ClassifierA(
                    n_input= N_LATENT,
                    n_output=N_LABELS,
                    do_batch_norm=False,
                    dropout_rate=0.1,
                )
            }
        )

if os.path.exists(PATH_C):
    print("model exists; loadizng from .pt")
    classifier.load_state_dict(torch.load(PATH_C))
classifier.eval()



LABELLED_PROPORTIONS = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 0.0])
LABELLED_PROPORTIONS = LABELLED_PROPORTIONS / LABELLED_PROPORTIONS.sum()

LABELLED_FRACTION = 0.5

DATASET = TrentoDataset(
    labelled_proportions=LABELLED_PROPORTIONS,
    labelled_fraction=LABELLED_FRACTION
)
x,y = DATASET.test_dataset.tensors # 29102
print(x.shape, y.shape, torch.unique(y))
print(torch.mean(x, dim = 0), torch.std(x, dim =0))

z1 = encoder["default"](x, n_samples= 1)["latent"]
qc_z1 = classifier["default"](z1)

cat_dist = RelaxedOneHotCategorical(
    temperature=0.5, probs=qc_z1
    )
ys_probs = cat_dist.rsample()
ys = (ys_probs == ys_probs.max(-1, keepdim=True).values).float()
y_int = ys.argmax(-1)

print(y_int.shape)

score = balanced_accuracy_score(y, y_int)
print(score)

score = accuracy_score(y, y_int)
print(score)