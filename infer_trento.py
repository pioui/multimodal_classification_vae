from email.policy import default
from mcvae.models.regular_modules import ClassifierA, EncoderB
from mcvae.models.semi_supervised_vae_relaxed import RelaxedSVAE
import tifffile

import torch
import os
from torch import nn
from scipy import io



def pixelwise_reshape(input):
    """
    compress
    """
    if len(input.shape)==3:
        return input.reshape(-1,input.shape[0])
    if len(input.shape)==2:
        return input.reshape(-1)

PATH_C =  "/home/pigi/repos/multimodal-decision-making-vaes/models/trento-relaxed_nparticules_30/classifier_7448361527406774815.pt"
PATH_E =  "/home/pigi/repos/multimodal-decision-making-vaes/models/trento-relaxed_nparticules_30/encoder_z1_7448361527406774815.pt"
PATH_M = "/home/pigi/repos/multimodal-decision-making-vaes/models/trento-relaxed_nparticules_30/10_0_ELBO_ELBO_None_30_30_True_10_128_vanilla_100_None_None_0.001_False_False_VAE_.pt"

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

x = torch.rand(1, 65)

z1= encoder['default'](x, N_SAMPLES)
print(x.shape,z1['latent'].shape)

y_pred= classifier['default'](z1['latent'])
print(y_pred.shape)

y_pred = mdl.classify(x, N_SAMPLES)
print(y_pred.shape)

data_dir = "/home/pigi/Documents/UiT_Internship/Trento/Trento/"

image_hyper = pixelwise_reshape(torch.tensor(tifffile.imread(data_dir+"hyper_Italy.tif"))) # 99600,63
image_lidar = pixelwise_reshape(torch.tensor(tifffile.imread(data_dir+"LiDAR_Italy.tif"))) # 99600,2
x = torch.cat((image_hyper,image_lidar), dim = 1) # 99600,65

y = io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"]
y = pixelwise_reshape(torch.tensor(y, dtype = torch.int64)) # 99600


rdm_idx = torch.randint(0,99600, (50000,))
y_pred_prob = mdl.classify(x[rdm_idx], N_SAMPLES)
y_pred = torch.argmax(y_pred_prob, dim = 1)
print(y_pred.shape, y[rdm_idx].shape)
print(sum(y_pred == y[rdm_idx]))

z1= encoder['default'](x[rdm_idx], 1)
y_pred_prob= classifier['default'](z1['latent'])
y_pred = torch.argmax(y_pred_prob, dim = 1)
print(y_pred.shape, y[rdm_idx].shape)
print(sum(y_pred == y[rdm_idx]))


