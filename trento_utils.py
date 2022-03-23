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

logging.basicConfig(filename = 'trento_logs.log',level=logging.DEBUG)

device = "cuda" if torch.cuda.is_available() else "cpu"


N_EVAL_SAMPLES = 25
NUM = 300
N_EXPERIMENTS = 5
LABELLED_PROPORTIONS = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 0.0])
LABELLED_PROPORTIONS = LABELLED_PROPORTIONS / LABELLED_PROPORTIONS.sum()
LABELLED_FRACTION = 0.5
np.random.seed(42)
N_INPUT = 65
N_LABELS = 5

CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 25
N_EPOCHS = 2
LR = 3e-4
BATCH_SIZE = 512
DATASET = TrentoDataset(
    labelled_fraction=LABELLED_FRACTION,
    labelled_proportions=LABELLED_PROPORTIONS,
    do_1d=True,
    test_size=0.5,
)
X_TRAIN, Y_TRAIN = DATASET.train_dataset.tensors
RDM_INDICES = np.random.choice(len(X_TRAIN), 200)
X_SAMPLE = X_TRAIN[RDM_INDICES].to(device)
Y_SAMPLE = Y_TRAIN[RDM_INDICES].to(device)
DO_OVERALL = True

# Utils functions
def compute_reject_score(y_true: np.ndarray, y_pred: np.ndarray, num=20):
    """
        Computes precision recall properties for the discovery label using
        Bayesian decision theory
    """
    _, n_pos_classes = y_pred.shape

    # assert np.unique(y_true).max() == (n_pos_classes - 1) + 1
    thetas = np.linspace(0.1, 1.0, num=num)
    properties = dict(
        precision_discovery=np.zeros(num),
        recall_discovery=np.zeros(num),
        accuracy=np.zeros(num),
        thresholds=thetas,
    )

    for idx, theta in enumerate(thetas):
        y_pred_theta = y_pred.argmax(1)
        reject = y_pred.max(1) <= theta
        y_pred_theta[reject] = (n_pos_classes - 1) + 1

        properties["accuracy"][idx] = accuracy_score(y_true, y_pred_theta)

        y_true_discovery = y_true == (n_pos_classes - 1) + 1
        y_pred_discovery = y_pred_theta == (n_pos_classes - 1) + 1
        properties["precision_discovery"][idx] = precision_score(
            y_true_discovery, y_pred_discovery, zero_division=0
        )
        properties["recall_discovery"][idx] = recall_score(
            y_true_discovery, y_pred_discovery, zero_division=0
        )
    return properties


def res_eval_loop(
    trainer,
    eval_encoder,
    counts_eval,
    encoder_eval_name,
    do_defensive: bool = False,
    debug: bool = False,
):
    model = trainer.model


    logging.info("Train Predictions computation ...")
    with torch.no_grad():
        # Below function integrates both inference methods for
        # mixture and simple statistics
        train_res = trainer.inference(
            # trainer.test_loader,
            trainer.train_annotated_loader,
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

    y_pred_is = train_res["preds_is"].numpy()
    # y_pred_is = y_pred_is / y_pred_is.sum(1, keepdims=True)
    assert y_pred.shape == y_pred_is.shape, (y_pred.shape, y_pred_is.shape)

    y_true = train_res["y"].numpy()

    # Precision / Recall for discovery class
    # And accuracy
    logging.info("Precision, recall, auc ...")
    res_baseline = compute_reject_score(y_true=y_true, y_pred=y_pred)
    train_m_ap = res_baseline["precision_discovery"]
    train_m_recall = res_baseline["recall_discovery"]
    train_auc_pr = np.trapz(
        x=res_baseline["recall_discovery"], y=res_baseline["precision_discovery"],
    )

    res_baseline_is = compute_reject_score(y_true=y_true, y_pred=y_pred_is)
    train_m_ap_is = res_baseline_is["precision_discovery"]
    train_m_recall_is = res_baseline_is["recall_discovery"]
    train_auc_pr_is = np.trapz(
        x=res_baseline_is["recall_discovery"], y=res_baseline_is["precision_discovery"],
    )

    # Entropy
    where9 = train_res["y"] == 5
    probas9 = train_res["qc_z1_all_probas"].mean(0)[where9]
    train_entropy = (-probas9 * probas9.log()).sum(-1).mean(0)

    where_non9 = train_res["y"] != 5
    y_non9 = train_res["y"][where_non9]
    y_pred_non9 = y_pred[where_non9].argmax(1)
    train_m_accuracy = accuracy_score(y_non9, y_pred_non9)

    y_pred_non9_is = y_pred_is[where_non9].argmax(1)
    train_m_accuracy_is = accuracy_score(y_non9, y_pred_non9_is)


    logging.info("Test Predictions computation ...")
    with torch.no_grad():
        # Below function integrates both inference methods for
        # mixture and simple statistics
        test_res = trainer.inference(
            trainer.test_loader,
            # trainer.train_loader,
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
    y_pred = test_res["preds_plugin"].numpy()
    y_pred = y_pred / y_pred.sum(1, keepdims=True)

    y_pred_is = test_res["preds_is"].numpy()
    # y_pred_is = y_pred_is / y_pred_is.sum(1, keepdims=True)
    assert y_pred.shape == y_pred_is.shape, (y_pred.shape, y_pred_is.shape)

    y_true = test_res["y"].numpy()

    # Precision / Recall for discovery class
    # And accuracy
    logging.info("Precision, recall, auc ...")
    res_baseline = compute_reject_score(y_true=y_true, y_pred=y_pred)
    m_ap = res_baseline["precision_discovery"]
    m_recall = res_baseline["recall_discovery"]
    auc_pr = np.trapz(
        x=res_baseline["recall_discovery"], y=res_baseline["precision_discovery"],
    )

    res_baseline_is = compute_reject_score(y_true=y_true, y_pred=y_pred_is)
    m_ap_is = res_baseline_is["precision_discovery"]
    m_recall_is = res_baseline_is["recall_discovery"]
    auc_pr_is = np.trapz(
        x=res_baseline_is["recall_discovery"], y=res_baseline_is["precision_discovery"],
    )

    # Entropy
    where9 = test_res["y"] == 5
    probas9 = test_res["qc_z1_all_probas"].mean(0)[where9]
    entropy = (-probas9 * probas9.log()).sum(-1).mean(0)

    where_non9 = test_res["y"] != 5
    y_non9 = test_res["y"][where_non9]
    y_pred_non9 = y_pred[where_non9].argmax(1)
    m_accuracy = accuracy_score(y_non9, y_pred_non9)
    m_confusion_matrix = confusion_matrix(y_non9, y_pred_non9)

    y_pred_non9_is = y_pred_is[where_non9].argmax(1)
    m_accuracy_is = accuracy_score(y_non9, y_pred_non9_is)
    m_confusion_matrix_is = confusion_matrix(y_non9, y_pred_non9_is)

    res = {
        # "IWELBO": iwelbo_vals.mean().item(),
        # "IWELBOC": iwelbo_c_vals.mean().item(),
        # "CUBO": cubo_vals.mean().item(),
        # "KHAT": np.array(khats),
        "M_ACCURACY": m_accuracy,
        "MEAN_AP": m_ap,
        "MEAN_RECALL": m_recall,
        # "KHATS_C_OBS": khats_c_obs,
        "M_ACCURACY_IS": m_accuracy_is,
        "MEAN_AP_IS": m_ap_is,
        "MEAN_RECALL_IS": m_recall_is,
        "AUC_IS": auc_pr_is,
        "AUC": auc_pr,
        "ENTROPY": entropy,
        "CONFUSION_MATRIX": m_confusion_matrix,
        "CONFUSION_MATRIX_IS": m_confusion_matrix_is,

        "train_M_ACCURACY": train_m_accuracy,
        "train_MEAN_AP": train_m_ap,
        "train_MEAN_RECALL": train_m_recall,
        "train_M_ACCURACY_IS": train_m_accuracy_is,
        "train_MEAN_AP_IS": train_m_ap_is,
        "train_MEAN_RECALL_IS": train_m_recall_is,
        "train_AUC_IS": train_auc_pr_is,
        "train_AUC": train_auc_pr,
        "train_ENTROPY": train_entropy,
        "train_LOSS": trainer.train_loss,
    }
    return res
