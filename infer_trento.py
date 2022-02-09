from email.policy import default
from mcvae.models.regular_modules import ClassifierA, EncoderB
from mcvae.models.semi_supervised_vae_relaxed import RelaxedSVAE
import tifffile

import torch
import os
from torch import nn
from scipy import io

from sklearn.metrics import classification_report, confusion_matrix
from trento_utils import compute_reject_score
from mcvae.dataset import TrentoDataset


def pixelwise_reshape(input):
    """
    compress
    """
    if len(input.shape)==3:
        return input.reshape(-1,input.shape[0])
    if len(input.shape)==2:
        return input.reshape(-1)

PATH_C =  "/home/pigi/repos/multimodal_classification_vae/models/trento-relaxed_nparticules_30/classifier_-3118629562734959505_epoch_850.pt"
PATH_E =  "/home/pigi/repos/multimodal_classification_vae/models/trento-relaxed_nparticules_30/encoder_z1_-3118629562734959505_epoch_850.pt"
PATH_M = "/home/pigi/repos/multimodal_classification_vae/models/trento-relaxed_nparticules_30/10_0_ELBO_ELBO_None_30_30_True_10_128_vanilla_1000_None_None_0.001_False_False_VAE_epoch_850.pt"

N_INPUT = 65
N_LABELS = 5 #one label is excluded
N_LATENT = 10
N_HIDDEN = 128
N_SAMPLES = 20

multi_encoder_keys: list = ["default"]


encoder =  nn.ModuleDict(
                {

                    key: EncoderB(
                        n_input=N_INPUT,
                        n_output=N_LATENT,
                        n_hidden=N_HIDDEN,
                        dropout_rate=0.1,
                        do_batch_norm=False,
                    )
                    for key in multi_encoder_keys
                }
            )
if os.path.exists(PATH_E):
    print("model exists; loadizng from .pt")
    encoder.load_state_dict(torch.load(PATH_E))
encoder.eval()

classifier = nn.ModuleDict(
            {
                key: ClassifierA(
                    n_input= N_LATENT,
                    n_output=N_LABELS,
                    do_batch_norm=False,
                    dropout_rate=0.1,
                )
                for key in multi_encoder_keys
            }
        )

if os.path.exists(PATH_C):
    print("model exists; loadizng from .pt")
    classifier.load_state_dict(torch.load(PATH_C))
classifier.eval()

mdl = RelaxedSVAE(
    n_input=N_INPUT,
    n_labels=N_LABELS,
    n_latent=N_LATENT,
    n_hidden=N_HIDDEN,
    n_layers=1,
    # do_batch_norm=batch_norm,
    # multi_encoder_keys=multi_encoder_keys,
    # vdist_map=vdist_map_train,
)
if os.path.exists(PATH_M):
    print("model exists; loadizng from .pt")
    mdl.load_state_dict(torch.load(PATH_M))
mdl.eval()

DATASET = TrentoDataset()
x,y = DATASET.test_dataset.tensors # 29102
print(x.shape, y.shape, torch.unique(y))
print(torch.mean(x, dim = 0), torch.std(x, dim =0))


print(" ")
for i in range(1):
    y_pred_prob1 = mdl.classify(x[:10000], N_SAMPLES)
    y_pred1 = torch.argmax(y_pred_prob1, dim = 1)+1

    y_pred_prob2 = mdl.classify(x[10000:], N_SAMPLES)
    y_pred2 = torch.argmax(y_pred_prob2, dim = 1)+1

    print((sum(y_pred1 == y[:10000])+sum(y_pred2 == y[10000:]))/ N)
    # print(confusion_matrix(y_true = y, y_pred=y_pred))
print(torch.unique(y_pred1))
print(torch.unique(y_pred2))

compute_reject_score(y[:10000]-1, y_pred_prob1)
compute_reject_score(y[10000:]-1, y_pred_prob2)





# print(" ")
# for i in range(1):

#     z1= encoder['default'](x[:10000], 1)
#     y_pred_prob1= classifier['default'](z1['latent'])
#     y_pred1 = torch.argmax(y_pred_prob1, dim = 1)+1

#     z1= encoder['default'](x[10000:], 1)
#     y_pred_prob2= classifier['default'](z1['latent'])
#     y_pred2 = torch.argmax(y_pred_prob2, dim = 1)+1

#     print((sum(y_pred1 == y[:10000])+sum(y_pred2 == y[10000:]))/ N)
#     print(confusion_matrix(y_true = y[:10000], y_pred=y_pred1))
# print(torch.unique(y_pred1))
# print(torch.unique(y_pred2))