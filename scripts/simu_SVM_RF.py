"""
Script to train and make predictions on SVM and RF classifiers
Usage:
  python3 scripts/simu_SVM_RF.py -d <DATASET NAME> 

"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

from mcvae.utils import generate_latex_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", "-d",
    help="name of dataset to use (trento, houston)",
    )

args = parser.parse_args()
dataset = args.dataset

if dataset == "trento":
    from trento_config import *
    from mcvae.dataset import trentoDataset
    DATASET = trentoDataset(data_dir=data_dir)
elif dataset == "houston":
    from houston_config import *
    from mcvae.dataset import houstonDataset
    DATASET = houstonDataset(data_dir=data_dir, samples_per_class=SAMPLES_PER_CLASS)

X_train,y_train = DATASET.train_dataset_labelled.tensors 
X_test,y_test = DATASET.test_dataset_labelled.tensors 
X,y = DATASET.full_dataset.tensors 

# ----- SVM -----#
print("----- SVM -----")
# Fit SVM classifier
clf_svm = SVC(
    C=1,
    kernel="rbf", 
    probability=True
    )
clf_svm.fit(X_train.numpy(), y_train.numpy())
# Calaculate metrics on test set
print("Metrics on test set")
y_pred = clf_svm.predict(X_test.numpy())
print("Accuracy: ", clf_svm.score(X_test.numpy(), y_test.numpy()))
print("F1-score weighted:", f1_score(y_test.numpy(), y_pred, average='weighted'))
print("Kappa score:", cohen_kappa_score(y_test.numpy(), y_pred))
print()
m_confusion_matrix = confusion_matrix(y_test, y_pred, normalize='true')
m_confusion_matrix = np.around(m_confusion_matrix.astype('float') / m_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
print(generate_latex_confusion_matrix(m_confusion_matrix))
            
# Classification map on the whole dataset
y_pred = clf_svm.predict(X.numpy())
plt.figure(dpi=500)
plt.imshow(y_pred.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color[1:]))
plt.axis('off')
plt.savefig(f"{images_dir}{dataset}_SVM_PREDICTIONS.png",bbox_inches='tight', pad_inches=0, dpi=500)

y_pred_prob = clf_svm.predict_proba(X.numpy())
np.save(f"{outputs_dir}{PROJECT_NAME}_SVM.npy", y_pred_prob)

# ----- RF -----
print(" ----- RF -----")
# Fit RF classifier
clf_rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    max_depth=80,
    min_samples_split=5,
    bootstrap=True,
    max_features="sqrt",
    )
clf_rf.fit(X_train.numpy(), y_train.numpy())
# Calaculate metrics on test set
print("Metrics on test set")
y_pred = clf_rf.predict(X_test.numpy())
print("Accuracy: ", clf_rf.score(X_test.numpy(), y_test.numpy()))
print("F1-score weighted:", f1_score(y_test.numpy(), y_pred, average='weighted'))
print("Kappa score:", cohen_kappa_score(y_test.numpy(), y_pred))
print()
m_confusion_matrix = confusion_matrix(y_test, y_pred, normalize='true')
m_confusion_matrix = np.around(m_confusion_matrix.astype('float') / m_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
print(generate_latex_confusion_matrix(m_confusion_matrix))
            
# Classification map on the whole dataset
y_pred = clf_rf.predict(X.numpy())
plt.figure(dpi=500)
plt.imshow(y_pred.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color[1:]))
plt.axis('off')
plt.savefig(f"{images_dir}{dataset}_RF_PREDICTIONS.png",bbox_inches='tight', pad_inches=0, dpi=500)

y_pred_prob = clf_rf.predict_proba(X.numpy())
np.save(f"{outputs_dir}{PROJECT_NAME}_RF.npy", y_pred_prob)


# python3 scripts/simu_SVM_RF.py -d houston
# python3 scripts/simu_SVM_RF.py -d trento
