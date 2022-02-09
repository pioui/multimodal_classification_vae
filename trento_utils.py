import os
import logging
import torch
import numpy as np
from arviz.stats import psislw
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm.auto import tqdm

from mcvae.dataset import TrentoDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# !! NOT ALL OF THE ARE ACTUALLY USED: #No!! 
LABELLED_FRACTION = 0.6
N_INPUT = 65
N_LABELS = 5 #one label is excluded

CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 25
N_EPOCHS = 1000
LR = 3e-4

BATCH_SIZE = 10
LABELLED_FRACTION = 0.6

DATASET = TrentoDataset(
    labelled_fraction=LABELLED_FRACTION,
)

#Used for evaluation
X_TRAIN, Y_TRAIN = DATASET.train_dataset.tensors 
RDM_INDICES = np.random.choice(len(X_TRAIN), 200) 
X_SAMPLE = X_TRAIN[RDM_INDICES].to(device) 
Y_SAMPLE = Y_TRAIN[RDM_INDICES].to(device) 

N_INPUT = X_TRAIN.shape[-1]
N_LABELS = torch.unique(Y_TRAIN).shape[0] - 1 #one label is excluded

DO_OVERALL = True
EXCLUDED_LABEL = N_LABELS #the last label


# Utils functions
def compute_reject_score(y_true: np.ndarray, y_pred: np.ndarray, num=20):
    """
        Computes precision recall properties for the discovery label using
        Bayesian decision theory

        y_true: true labels (1,2,3,...,N)
        y_preds: prediction probabilities for N-1 classes
        num: number of probabilities between(0.1,1) that are used as thresholds 
             for rejecting a prediction and assigned it to the Nth label.
        
    """
    with torch.no_grad():
        _, n_pos_classes = y_pred.shape

        assert np.unique(y_true).max() == n_pos_classes
        
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
            y_pred_theta[reject] = n_pos_classes

            properties["accuracy"][idx] = accuracy_score(y_true, y_pred_theta)

            y_true_discovery = y_true == n_pos_classes
            y_pred_discovery = y_pred_theta == n_pos_classes
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

    logging.info("Predictions computation ...")
    with torch.no_grad():
        # Below function integrates both inference methods for
        # mixture and simple statistics
        train_res = trainer.inference(
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

    # Cubo / Iwelbo with n_samples_total samples
    logging.info("Heldout CUBO/IWELBO computation ...")

    n_samples_total = 500 # Samples used 
    if debug:
        n_samples_total = 200
    n_samples_per_pass = 25 if not do_defensive else counts_eval.sum()
    n_iter = int(n_samples_total / n_samples_per_pass)

    cubo_vals = []
    iwelbo_vals = []
    iwelbo_c_vals = []

    with torch.no_grad():
        i = 0
        for tensors in tqdm(trainer.test_loader):
            x, _ = tensors
            x = x.to(device)
            log_ratios_batch = []
            log_qc_batch = []
            for _ in tqdm(range(n_iter)):
                out = model.inference(
                    x,
                    temperature=0.5,
                    n_samples=n_samples_per_pass,
                    encoder_key=encoder_eval_name,
                    counts=counts_eval,
                )
                if do_defensive:
                    log_ratio = out["log_ratio"].cpu()
                else:
                    log_ratio = (
                        out["log_px_z"]
                        + out["log_pz2"]
                        + out["log_pc"]
                        + out["log_pz1_z2"]
                        - out["log_qz1_x"]
                        - out["log_qc_z1"]
                        - out["log_qz2_z1"]
                    ).cpu()
                log_ratios_batch.append(log_ratio)
                log_qc_batch.append(out["log_qc_z1"].cpu())

            i += 1
            if i == 20:
                break
            # Concatenation
            log_ratios_batch = torch.cat(log_ratios_batch, dim=1)
            log_qc_batch = torch.cat(log_qc_batch, dim=1)

            # Lower bounds
            # 1. Cubo
            # n_cat, n_samples, n_batch = log_ratios_batch.shape
            # cubo_val = torch.logsumexp(
            #     (2 * log_ratios_batch + log_qc_batch).view(n_cat * n_samples, n_batch),
            #     dim=0,
            #     keepdim=False,
            # ) - np.log(n_samples)

            # iwelbo_val = torch.logsumexp(
            #     (log_ratios_batch + log_qc_batch).view(n_cat * n_samples, n_batch),
            #     dim=0,
            #     keepdim=False,
            # ) - np.log(n_samples)
            # IWELBO C
            # # n_cat, n_samples, n_batch
            # qc_probs = log_qc_batch.permute([1, 2, 0]).exp()
            # qc_dist = Categorical(probs=qc_probs)
            # c_sampled = qc_dist.sample().unsqueeze(0)
            # # log_qc_samp = torch.gather(log_qc_batch, dim=0, index=c_sampled)
            # log_ratios_samp = torch.gather(log_ratios_batch, dim=0, index=c_sampled)
            # # Shape 1, n_samples, n_batch
            # iwelboc_val = torch.logsumexp(log_ratios_samp, dim=1) - np.log(n_samples)
            # iwelboc_val = iwelboc_val.squeeze()

            # cubo_vals.append(cubo_val.cpu())
            # iwelbo_vals.append(iwelbo_val.cpu())
            # iwelbo_c_vals.append(iwelboc_val.cpu())

            # RELAXED CASE
            n_samples, n_batch = log_ratios_batch.shape
            cubo_val = torch.logsumexp(
                2 * log_ratios_batch, dim=0, keepdim=False,
            ) - np.log(n_samples)

            iwelbo_val = torch.logsumexp(
                log_ratios_batch, dim=0, keepdim=False,
            ) - np.log(n_samples)

            cubo_vals.append(cubo_val.cpu())
            iwelbo_vals.append(iwelbo_val.cpu())
        cubo_vals = torch.cat(cubo_vals)
        iwelbo_vals = torch.cat(iwelbo_vals)
        # iwelbo_c_vals = torch.cat(iwelbo_c_vals)

    # Entropy
    where9 = train_res["y"] == EXCLUDED_LABEL
    probas9 = train_res["qc_z1_all_probas"].mean(0)[where9]
    entropy = (-probas9 * probas9.log()).sum(-1).mean(0)

    where_non9 = train_res["y"] != EXCLUDED_LABEL
    y_non9 = train_res["y"][where_non9]
    y_pred_non9 = y_pred[where_non9].argmax(1)
    m_accuracy = accuracy_score(y_non9, y_pred_non9)

    y_pred_non9_is = y_pred_is[where_non9].argmax(1)
    m_accuracy_is = accuracy_score(y_non9, y_pred_non9_is)

    # k_hat

    # n_samples_total = 1000
    # if debug:
    #     n_samples_total = 200
    # n_samples_per_pass = 100 if not do_defensive else counts_eval.sum()
    # n_iter = int(n_samples_total / n_samples_per_pass)

    # a. Unsupervised case
    # log_ratios = []
    # qc_z = []
    # for _ in tqdm(range(n_iter)):
    #     with torch.no_grad():
    #         out = model.inference(
    #             X_SAMPLE,
    #             temperature=0.5,
    #             n_samples=n_samples_per_pass,
    #             encoder_key=encoder_eval_name,
    #             counts=counts_eval,
    #         )
    #     if do_defensive:
    #         log_ratio = out["log_ratio"].cpu()
    #     else:
    #         log_ratio = (
    #             out["log_px_z"]
    #             + out["log_pz2"]
    #             + out["log_pc"]
    #             + out["log_pz1_z2"]
    #             - out["log_qz1_x"]
    #             - out["log_qc_z1"]
    #             - out["log_qz2_z1"]
    #         ).cpu()
    #     qc_z_here = out["log_qc_z1"].cpu().exp()
    #     qc_z.append(qc_z_here)
    #     log_ratios.append(log_ratio)
    # # Concatenation over samples
    # log_ratios = torch.cat(log_ratios, 1)
    # qc_z = torch.cat(qc_z, 1)
    # log_ratios_sum = (log_ratios * qc_z).sum(0)  # Sum over labels
    # wi = torch.softmax(log_ratios_sum, 0)
    # _, khats = psislw(log_ratios_sum.T.clone())

    log_ratios = []
    for _ in tqdm(range(n_iter)):
        with torch.no_grad():
            out = model.inference(
                X_SAMPLE,
                temperature=0.5,
                n_samples=n_samples_per_pass,
                encoder_key=encoder_eval_name,
                counts=counts_eval,
            )
        if do_defensive:
            log_ratio = out["log_ratio"].cpu()
        else:
            log_ratio = (
                out["log_px_z"]
                + out["log_pz2"]
                + out["log_pc"]
                + out["log_pz1_z2"]
                - out["log_qz1_x"]
                - out["log_qc_z1"]
                - out["log_qz2_z1"]
            ).cpu()
        log_ratios.append(log_ratio)
    # Concatenation over samples
    log_ratios = torch.cat(log_ratios, 0)
    _, khats = psislw(log_ratios.T.clone())

    x_samp, y_samp = DATASET.train_dataset[:128]
    where_ = y_samp != EXCLUDED_LABEL
    x_samp = x_samp[where_].to(device)
    y_samp = y_samp[where_].to(device)
    log_ratios = []
    for _ in tqdm(range(n_iter)):
        with torch.no_grad():
            out = model.inference(
                x_samp,
                y_samp,
                temperature=0.5,
                n_samples=n_samples_per_pass,
                encoder_key=encoder_eval_name,
                counts=counts_eval,
            )
        if do_defensive:
            log_ratio = out["log_ratio"].cpu()
        else:
            log_ratio = (
                out["log_px_z"]
                + out["log_pz2"]
                + out["log_pc"]
                + out["log_pz1_z2"]
                - out["log_qz1_x"]
                # - out["log_qc_z1"]
                - out["log_qz2_z1"]
            ).cpu()
        log_ratios.append(log_ratio)
    # Concatenation over samples
    log_ratios = torch.cat(log_ratios, 0)
    _, khats_c_obs = psislw(log_ratios.T.clone())

    res = {
        "IWELBO": iwelbo_vals.mean().item(),
        # "IWELBOC": iwelbo_c_vals.mean().item(),
        "CUBO": cubo_vals.mean().item(),
        "KHAT": np.array(khats),
        "M_ACCURACY": m_accuracy,
        "MEAN_AP": m_ap,
        "MEAN_RECALL": m_recall,
        "KHATS_C_OBS": khats_c_obs,
        "M_ACCURACY_IS": m_accuracy_is,
        "MEAN_AP_IS": m_ap_is,
        "MEAN_RECALL_IS": m_recall_is,
        "AUC_IS": auc_pr_is,
        "AUC": auc_pr,
        "ENTROPY": entropy,
    }
    return res




