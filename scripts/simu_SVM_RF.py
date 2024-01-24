"""
This script is specifically designed to handle the training and prediction tasks 
associated with SVM and RF architectures.
Usage:
  python3 scripts/simu_SVM_RF.py -d <DATASET NAME> 

Replace <DATASET NAME> with the specific dataset you intend to use. The script will 
then initiate the training process for the M1+M2 model using the specified dataset. 
Once the training is complete, the script allows you to make accurate predictions 
on new data.
"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score,classification_report, accuracy_score
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", "-d",
    help="name of dataset to use (trento, houston)",
    )

args = parser.parse_args()
dataset = args.dataset

if dataset == "trento":
    from trento_config import *
    from mcvae.dataset import trento_dataset

    C = 100
    kernel = 'rbf'
    gamma = 'scale'

    n_estimators=300
    criterion = 'gini'
    min_samples_leaf=2
    max_depth=80
    min_samples_split=5
    bootstrap=True
    max_features="sqrt"
    n_jobs = 5

    DATASET = trento_dataset(data_dir=data_dir)
elif dataset == "houston":
    from houston_config import *
    from mcvae.dataset import houston_dataset
    DATASET = houston_dataset(data_dir=data_dir, samples_per_class=SAMPLES_PER_CLASS)

    C = 100
    kernel = 'rbf'
    gamma = 'scale'

    n_estimators=300
    criterion = 'gini'
    min_samples_leaf=2
    max_depth=80
    min_samples_split=5
    bootstrap=True
    max_features="sqrt"
    n_jobs = 5


else:
    print("Dataset name is not valid. Please try one of the following: trento, houston, trento-patch, houston-patch")
    exit()

X_train,y_train = DATASET.train_dataset_labelled.tensors 
X_test,y_test = DATASET.test_dataset_labelled.tensors 
X,y = DATASET.full_dataset.tensors 

# ----- SVM -----#
print("Fitting SVM...")
# Fit SVM classifier
clf_svm = SVC(
    C=C,
    kernel="rbf", 
    gamma = gamma,
    probability=True
    )
start_time = time.time()
clf_svm.fit(X_train.numpy(), y_train.numpy())
end_time = time.time()
elapsed_time = end_time - start_time
print("Training Elapsed time: ", elapsed_time) 

start_time = time.time()
y_pred_prob = clf_svm.predict_proba(X.numpy())
end_time = time.time()
elapsed_time = end_time - start_time
print("Testing Elapsed time: ", elapsed_time) 

np.save(f"{outputs_dir}{PROJECT_NAME}_SVM.npy", y_pred_prob)

y_pred = clf_svm.predict(X_test.numpy())
print("Accuracy:", classification_report(y_pred,y_test))



# ----- RF -----
print("Fitting RF...")
# Fit RF classifier
clf_rf = RandomForestClassifier(
    n_estimators=n_estimators,
    criterion=criterion,
    min_samples_leaf=min_samples_leaf,
    max_depth=max_depth,
    bootstrap=bootstrap,
    max_features=max_features,
    n_jobs=n_jobs
    )
start_time = time.time()
clf_rf.fit(X_train.numpy(), y_train.numpy())
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time) 
start_time = time.time()
y_pred_prob = clf_rf.predict_proba(X.numpy())
end_time = time.time()
elapsed_time = end_time - start_time
print("Testing Elapsed time: ", elapsed_time) 

np.save(f"{outputs_dir}{PROJECT_NAME}_RF.npy", y_pred_prob)

y_pred = clf_svm.predict(X_test.numpy())
print("Accuracy:", classification_report(y_pred,y_test))