"""
    Decision theory: Experiment for M1+M1 model on trento
"""

import os
import logging
from math import ceil
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

from sklearn.metrics import confusion_matrix
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from arviz.stats import psislw
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm.auto import tqdm

from mcvae.architectures import VAE_M1M2
from mcvae.inference import VAE_M1M2_Trainer
from mcvae.architectures.regular_modules import (
    EncoderA,
    EncoderB,
    ClassifierA,
    EncoderAStudent,
    EncoderBStudent,
)

from trento_config import (
    outputs_dir,
    N_PARTICULES,
    N_LATENT,
    N_EPOCHS,
    N_HIDDEN,
    LR,
    N_EXPERIMENTS,
    BATCH_SIZE,
    CLASSIFICATION_RATIO,
    N_EVAL_SAMPLES,
    N_INPUT,
    N_LABELS,
    DATASET,
    PROJECT_NAME,
    SCENARIOS,
)
from utility_functions import (
    model_evaluation,
)



device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_MAP = dict(
    REVKL="gaussian",
    CUBO="student",
    ELBO="gaussian",
    IWELBO="gaussian",
    IWELBOC="gaussian",
    default="gaussian",
)

Z1_MAP = dict(gaussian=EncoderB, student=EncoderBStudent,)
Z2_MAP = dict(gaussian=EncoderA, student=EncoderAStudent,)

FILENAME = f"{outputs_dir}/{PROJECT_NAME}.pkl"
MDL_DIR = f"models/{PROJECT_NAME}"
DEBUG = False

if not os.path.exists(MDL_DIR):
    os.makedirs(MDL_DIR)
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


logger.info("train all examples {}".format(len(DATASET.train_dataset.tensors[0])))
logger.info("train labelled examples {}".format(len(DATASET.train_dataset_labelled.tensors[0])))
logger.info("test labelled examples {}".format(len(DATASET.test_dataset.tensors[0])))

EVAL_ENCODERS = [
    dict(encoder_type="train", eval_encoder_name="train"),  # MUST BE ON TOP!!!
    dict(encoder_type="ELBO", reparam=True, eval_encoder_name="VAE"),
]



DF_LI = []
logging.info("Number of experiments : {}".format(N_EXPERIMENTS))
# Main script
for scenario in SCENARIOS:
    loss_gen = scenario.get("loss_gen", None)
    loss_wvar = scenario.get("loss_wvar", None)
    n_samples_train = scenario.get("n_samples_train", None)
    n_samples_wtheta = scenario.get("n_samples_wtheta", N_PARTICULES)
    n_samples_wphi = scenario.get("n_samples_wphi", N_PARTICULES)
    reparam_latent = scenario.get("reparam_latent", None)
    n_epochs = scenario.get("n_epochs", N_EPOCHS)
    n_latent = scenario.get("n_latent", N_LATENT)
    n_hidden = scenario.get("n_hidden", N_HIDDEN)
    vdist_map_train = scenario.get("vdist_map", None)
    classify_mode = scenario.get("classify_mode", "vanilla")
    lr = scenario.get("lr", LR)
    z2_with_elbo = scenario.get("z2_with_elbo", False)
    counts = scenario.get("counts", None)
    model_name = scenario.get("model_name", None)

    batch_norm = scenario.get("batch_norm", False)
    cubo_z2_with_elbo = scenario.get("cubo_z2_with_elbo", False)
    batch_size = scenario.get("batch_size", BATCH_SIZE)

    encoder_z1=scenario.get("encoder_z1", None)

    do_defensive = type(loss_wvar) == list
    multi_encoder_keys = loss_wvar if do_defensive else ["default"]
    for t in range(N_EXPERIMENTS):
        # t = t + 4
        loop_setup_dict = {
            "BATCH_SIZE": BATCH_SIZE,
            "ITER": t,
            "LOSS_GEN": loss_gen,
            "LOSS_WVAR": loss_wvar,
            "N_SAMPLES_TRAIN": n_samples_train,
            "N_SAMPLES_WTHETA": n_samples_wtheta,
            "N_SAMPLES_WPHI": n_samples_wphi,
            "REPARAM_LATENT": reparam_latent,
            "N_LATENT": n_latent,
            "N_HIDDEN": n_hidden,
            "CLASSIFY_MODE": classify_mode,
            "N_EPOCHS": n_epochs,
            "COUNTS": counts,
            "VDIST_MAP_TRAIN": vdist_map_train,
            "LR": lr,
            "BATCH_NORM": batch_norm,
            "Z2_WITH_ELBO": z2_with_elbo,
            "MODEL_NAME": model_name,
        }

        scenario["num"] = t
        mdl_name = ""
        for st in loop_setup_dict.values():
            mdl_name = mdl_name + str(st) + "_"
        mdl_name = str(mdl_name)
        mdl_name = os.path.join(MDL_DIR, "{}.pt".format(mdl_name))
        logger.info(mdl_name)
        while True:
            try:
                mdl = VAE_M1M2(
                    n_input=N_INPUT,
                    n_labels=N_LABELS,
                    n_latent=n_latent,
                    n_hidden=n_hidden,
                    n_layers=1,
                    do_batch_norm=batch_norm,
                    multi_encoder_keys=multi_encoder_keys,
                    vdist_map=vdist_map_train,
                    encoder_z1=encoder_z1
                )
                if os.path.exists(mdl_name):
                    logger.info("model exists; loading from .pt")
                    mdl.load_state_dict(torch.load(mdl_name))
                mdl.to(device)
                trainer = VAE_M1M2_Trainer(
                    dataset=DATASET,
                    model=mdl,
                    use_cuda=True,
                    batch_size=batch_size,
                    classify_mode=classify_mode,
                )
                overall_loss = None
                if not os.path.exists(mdl_name):
                    if do_defensive:
                        trainer.train_defensive(
                            n_epochs=n_epochs,
                            lr=lr,
                            wake_theta=loss_gen,
                            n_samples_phi=n_samples_wphi,
                            n_samples_theta=n_samples_wtheta,
                            classification_ratio=CLASSIFICATION_RATIO,
                            update_mode="all",
                            counts=counts,
                        )
                    else:
                        trainer.train(
                            n_epochs=n_epochs,
                            lr=lr,
                            overall_loss=overall_loss,
                            wake_theta=loss_gen,
                            wake_psi=loss_wvar,
                            n_samples=n_samples_train,
                            n_samples_theta=n_samples_wtheta,
                            n_samples_phi=n_samples_wphi,
                            reparam_wphi=reparam_latent,
                            classification_ratio=CLASSIFICATION_RATIO,
                            z2_with_elbo=z2_with_elbo,
                            update_mode="all",
                        )
                    
            except ValueError as e:
                logger.info(e)
                continue
            break
        torch.save(mdl.state_dict(), mdl_name)

        with torch.no_grad():
            train_res = trainer.inference(
                trainer.full_loader,
                keys=[
                    "qc_z1_all_probas",
                    "y",
                    "log_ratios",
                    "qc_z1",
                    "preds_is",
                    "preds_plugin",
                ],
                n_samples=N_EVAL_SAMPLES,
            )
        y_pred = train_res["preds_plugin"].numpy()
        y_pred = y_pred / y_pred.sum(1, keepdims=True)
        np.save(f"{outputs_dir}{model_name}.npy", y_pred)



        mdl.eval()
        # TODO: find something cleaner
        if do_defensive:
            factor = N_EVAL_SAMPLES / counts.sum()
            multi_counts = factor * counts
            multi_counts = multi_counts.astype(int)
        else:
            multi_counts = None

        for eval_dic in EVAL_ENCODERS:
            encoder_type = eval_dic.get("encoder_type", None)
            eval_encoder_name = eval_dic.get("eval_encoder_name", None)
            reparam = eval_dic.get("reparam", None)
            counts_eval = eval_dic.get("counts_eval", None)
            vdist_map_eval = eval_dic.get("vdist_map", DEFAULT_MAP)
            eval_encoder_loop = {
                "encoder_type": encoder_type,
                "eval_encoder_name": eval_encoder_name,
                "reparam_eval": reparam,
                "counts_eval": counts_eval,
                "vdist_map_eval": vdist_map_eval,
            }
            logger.info("ENCODER TYPE : {}".format(encoder_type))
            if encoder_type == "train":
                logging.info("Using train variational distribution for evaluation ...")
                eval_encoder = None
                do_defensive_eval = do_defensive
                multi_counts_eval = multi_counts
            else:
                logging.info(
                    "Training eval variational distribution for evaluation with {} ...".format(
                        encoder_type
                    )
                )
                do_defensive_eval = type(encoder_type) == list
                multi_encoder_keys_eval = (
                    encoder_type if do_defensive_eval else ["default"]
                )
                encoder_eval_name = None if do_defensive_eval else "default"
                if counts_eval is not None:
                    multi_counts_eval = 12 * counts_eval
                else:
                    multi_counts_eval = None

                while True:
                    try:
                        logging.info("Using map {} ...".format(vdist_map_eval))
                        new_classifier = nn.ModuleDict(
                            {
                                key: ClassifierA(
                                    n_latent,
                                    n_output=N_LABELS,
                                    do_batch_norm=False,
                                    dropout_rate=0.1,
                                )
                                for key in multi_encoder_keys_eval
                            }
                        ).to(device)
                        new_encoder_z1 = encoder_z1.to(device)

                        new_encoder_z2_z1 = nn.ModuleDict(
                            {
                                # key: EncoderA(
                                key: Z2_MAP[vdist_map_eval[key]](
                                    n_input=n_latent + N_LABELS,
                                    n_output=n_latent,
                                    n_hidden=n_hidden,
                                    dropout_rate=0.1,
                                    do_batch_norm=False,
                                )
                                for key in multi_encoder_keys_eval
                            }
                        ).to(device)
                        encoders = dict(
                            classifier=new_classifier,
                            encoder_z1=new_encoder_z1,
                            encoder_z2_z1=new_encoder_z2_z1,
                        )
                        all_dc = {**loop_setup_dict, **eval_encoder_loop}
                        eval_encoder_rootname = str(
                            hash(frozenset(pd.Series(all_dc).astype(str).items()))
                        )
                        mdl_names = {
                            key: os.path.join(
                                MDL_DIR,
                                "{}.pt".format(key + "_" + eval_encoder_rootname),
                            )
                            for key in encoders.keys()
                        }
                        filen_exists_arr = [
                            os.path.exists(filen) for filen in mdl_names.values()
                        ]
                        if np.array(filen_exists_arr).all():
                            logging.info("Loading eval mdls")
                            for key in mdl_names:
                                encoders[key].load_state_dict(
                                    torch.load(mdl_names[key])
                                )
                            mdl.update_q(**encoders)
                        else:
                            logging.info("training {}".format(encoder_type))
                            trainer.train_eval_encoder(
                                encoders=encoders,
                                n_epochs=n_epochs,
                                lr=lr,
                                wake_psi=encoder_type,
                                n_samples_phi=30,
                                classification_ratio=CLASSIFICATION_RATIO,
                                reparam_wphi=reparam,
                            )
                            for key in mdl_names:
                                torch.save(encoders[key].state_dict(), mdl_names[key])

                    except ValueError as e:
                        logger.info(e)
                        continue
                    break
                torch.save(mdl.state_dict(), mdl_name[:-3]+"train"+".pt")

            with torch.no_grad():
                train_res = trainer.inference(
                    trainer.full_loader,
                    keys=[
                        "qc_z1_all_probas",
                        "y",
                        "log_ratios",
                        "qc_z1",
                        "preds_is",
                        "preds_plugin",
                    ],
                    n_samples=N_EVAL_SAMPLES,
                )
            y_pred = train_res["preds_plugin"].numpy()
            y_pred = y_pred / y_pred.sum(1, keepdims=True)
            np.save(f"{outputs_dir}{model_name}_ELBO.npy", y_pred)

            logger.info(trainer.model.encoder_z2_z1.keys())
            loop_results_dict = model_evaluation(
                trainer=trainer,
                counts_eval=multi_counts_eval,
                encoder_eval_name="default",
            )
            res = {**loop_setup_dict, **loop_results_dict, **eval_encoder_loop}
            logger.info(res)
            DF_LI.append(res)
            DF = pd.DataFrame(DF_LI)
            DF.to_pickle(FILENAME)

# DF = pd.DataFrame(DF_LI)
# DF.to_pickle(FILENAME)
