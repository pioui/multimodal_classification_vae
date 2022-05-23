from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from sklearn.metrics import confusion_matrix

from mcvae.utils.utility_functions import variance_heterophil
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", "-d",
    help="name of dataset to use (trento, houston)",
    )

args = parser.parse_args()
dataset = args.dataset

if dataset=="trento":
    from trento_config import (
        outputs_dir,
        data_dir,
        N_PARTICULES,
        N_EPOCHS,
        N_HIDDEN,
        LR,
        N_EXPERIMENTS,
        BATCH_SIZE,
        CLASSIFICATION_RATIO,
        N_EVAL_SAMPLES,
        N_INPUT,
        N_LABELS,
        PROJECT_NAME,
        SCENARIOS,
        SHAPE,
        images_dir,
        labels,
        color,
        heterophil_matrix,

    )
    from mcvae.dataset import trentoDataset
    DATASET = trentoDataset(
        data_dir = data_dir,

    )
elif dataset=="houston":
    from houston_config import (
        outputs_dir,
        data_dir,
        N_PARTICULES,
        N_EPOCHS,
        N_HIDDEN,
        LR,
        N_EXPERIMENTS,
        BATCH_SIZE,
        CLASSIFICATION_RATIO,
        N_EVAL_SAMPLES,
        N_INPUT,
        N_LABELS,
        PROJECT_NAME,
        SCENARIOS,
        SAMPLES_PER_CLASS,
        SHAPE,
        images_dir,
        labels,
        color,
        heterophil_matrix

    )
    from mcvae.dataset import houstonDataset
    DATASET = houstonDataset(
        data_dir = data_dir,
        samples_per_class=SAMPLES_PER_CLASS,
    )

X_train,y_train = DATASET.train_dataset_labelled.tensors # 819
X_test,y_test = DATASET.test_dataset_labelled.tensors # 15107
X,y = DATASET.full_dataset.tensors # 15107

# ----- SVM -----#

clf_svm = SVC(C=1, kernel="rbf", probability=True)

clf_svm.fit(X_train.numpy(), y_train.numpy())

svm_accuracy = clf_svm.score(X_test.numpy(), y_test.numpy())
print(svm_accuracy)
y_pred = clf_svm.predict(X_test.numpy())
m_confusion_matrix = confusion_matrix(y_test, y_pred, normalize='true')
m_confusion_matrix = np.around(m_confusion_matrix.astype('float') / m_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
plt.figure(dpi=500)
plt.matshow(m_confusion_matrix, cmap="coolwarm")
plt.xlabel("True Labels")
plt.xticks(np.arange(0,N_LABELS,1), range(1,len(labels)))
plt.ylabel("Predicted Labels")
plt.yticks(np.arange(0,N_LABELS,1), range(1,len(labels)))
for k in range (len(m_confusion_matrix)):
    for l in range(len(m_confusion_matrix[k])):
        # plt.text(k,l,str(m_confusion_matrix[k][l]), va='center', ha='center', fontsize='small') #trento
        plt.text(k,l,str(m_confusion_matrix[k][l]), va='center', ha='center', fontsize='xx-small') #houston
plt.savefig(f"{images_dir}{dataset}_SVM_test_CONFUSION_MATRIX.png",bbox_inches='tight', pad_inches=0.2, dpi=500)
            
y_pred = clf_svm.predict(X.numpy())
plt.figure(dpi=500)
plt.imshow(y_pred.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color[1:]))
plt.axis('off')
plt.savefig(f"{images_dir}{dataset}_SVM_PREDICTIONS.png",bbox_inches='tight', pad_inches=0, dpi=500)

y_pred_prob = clf_svm.predict_proba(X.numpy())
plt.figure(dpi=500)
plt.imshow(variance_heterophil(p = y_pred_prob, w = heterophil_matrix ).reshape(SHAPE), cmap='coolwarm', 
# vmin=0, vmax=2.25
vmin=0, vmax=6.25
)
plt.axis('off')
cbar = plt.colorbar(location='top')
cbar.ax.tick_params(labelsize =8 )
plt.savefig(f"{images_dir}{dataset}_SVM_VARIANCE_HETERO.png",bbox_inches='tight', pad_inches=0.1 ,dpi=500)


np.save(f"{outputs_dir}{PROJECT_NAME}_SVM.npy", y_pred)


# # ----- RF -----

clf_rf = RandomForestClassifier(

n_estimators=300,

min_samples_leaf=2,

max_depth=80,

min_samples_split=5,

bootstrap=True,

max_features="sqrt",

)

clf_rf.fit(X_train, y_train)

svm_accuracy = clf_rf.score(X_test.numpy(), y_test.numpy())
print(svm_accuracy)
y_pred = clf_rf.predict(X_test.numpy())
m_confusion_matrix = confusion_matrix(y_test, y_pred, normalize='true')
m_confusion_matrix = np.around(m_confusion_matrix.astype('float') / m_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
plt.figure(dpi=500)
plt.matshow(m_confusion_matrix, cmap="coolwarm")
plt.xlabel("True Labels")
plt.xticks(np.arange(0,N_LABELS,1), range(1,len(labels)))
plt.ylabel("Predicted Labels")
plt.yticks(np.arange(0,N_LABELS,1), range(1,len(labels)))
for k in range (len(m_confusion_matrix)):
    for l in range(len(m_confusion_matrix[k])):
        # plt.text(k,l,str(m_confusion_matrix[k][l]), va='center', ha='center', fontsize='small') #trento
        plt.text(k,l,str(m_confusion_matrix[k][l]), va='center', ha='center', fontsize='xx-small') #houston
plt.savefig(f"{images_dir}{dataset}_RF_test_CONFUSION_MATRIX.png",bbox_inches='tight', pad_inches=0.2, dpi=500)
            

y_pred = clf_rf.predict(X.numpy())
plt.figure(dpi=500)
plt.imshow(y_pred.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color[1:]))
plt.axis('off')
plt.savefig(f"{images_dir}{dataset}_RF_PREDICTIONS.png",bbox_inches='tight', pad_inches=0, dpi=500)

y_pred_prob = clf_rf.predict_proba(X.numpy())
plt.figure(dpi=500)
plt.imshow(variance_heterophil(p = y_pred_prob, w = heterophil_matrix ).reshape(SHAPE), cmap='coolwarm', 
# vmin=0, vmax=2.25
vmin=0, vmax=6.25
)
plt.axis('off')
cbar = plt.colorbar(location='top')
cbar.ax.tick_params(labelsize =8 )
plt.savefig(f"{images_dir}{dataset}_RF_VARIANCE_HETERO.png",bbox_inches='tight', pad_inches=0.1 ,dpi=500)

np.save(f"{outputs_dir}{PROJECT_NAME}_RF.npy", y_pred)