import os
import logging
import torch
from torch.distributions import Categorical
import numpy as np
from arviz.stats import psislw
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm.auto import tqdm
import random
random.seed(42)
from mcvae.dataset import TrentoDataset
import matplotlib.pyplot as plt

from trento_config import (
    N_EVAL_SAMPLES,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(42)

DO_OVERALL = True

# Utils functions
def compute_reject_label(y_pred_prob, threshold):
    y_pred = y_pred_prob.argmax(1)+1

    max_prob = y_pred_prob.max(1)

    reject_idx = (max_prob<threshold)
    y_pred[reject_idx]=0

    return y_pred


def model_evaluation(
    trainer,
    counts_eval,
    encoder_eval_name,
):

    logger.info("Train Predictions computation ...")
    with torch.no_grad():
        train_res = trainer.inference(
            trainer.test_loader,
            keys=[
                "qc_z1_all_probas",
                "y",
                "log_ratios",
                "qc_z1",
                "preds_is",
                "preds_plugin",
            ],
            n_samples=N_EVAL_SAMPLES,
            encoder_key=encoder_eval_name,
            counts=counts_eval,
        )
    y_pred = train_res["preds_plugin"].numpy()
    y_pred = y_pred / y_pred.sum(1, keepdims=True)
    y_true = train_res["y"].numpy()

    logger.info("Precision, recall, auc, ...")
    m_precision = precision_score(y_true, y_pred.argmax(1), average = None, zero_division =0)
    m_recall = recall_score(y_true, y_pred.argmax(1), average = None, zero_division =0)
    m_accuracy = accuracy_score(y_true, y_pred.argmax(1))
    m_confusion_matrix = confusion_matrix(y_true+1, y_pred.argmax(1)+1)
    
    res = {
        "M_ACCURACY": m_accuracy,
        "MEAN_PRECISION": m_precision,
        "MEAN_RECALL": m_recall,
        "CONFUSION_MATRIX": m_confusion_matrix,
        "test_LOSS": trainer.test_loss,
        "train_LOSS": trainer.train_loss,
    }
    return res