from email.mime import image
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from scipy import io
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score
import os
import numpy as np
import torch
import tifffile
import csv
import pylab as pl


from mcvae.utils import generate_latex_matrix_from_dict, generate_latex_confusion_matrix

print(os.listdir('outputs/'))

for project_name in os.listdir('outputs/'):
    if project_name == 'trento':
        # continue
        dataset = 'trento'
        from trento_config import *
    elif project_name == 'trento_patch':
        # continue
        dataset = 'trento'
        from trento_patch_config import *
    elif project_name == 'trento_multimodal':
        # continue
        dataset = 'trento'
        from trento_multimodal_config import *

    elif project_name == 'houston':
        dataset = 'houston'
        from houston_config import *
    elif project_name == 'houston_patch':
        dataset = 'houston'
        from houston_patch_config import *
    elif project_name == 'houston_multimodal':
        dataset = 'houston'
        from houston_multimodal_config import *
    else:
        continue

    if dataset == "trento":
        from mcvae.dataset import trento_dataset
        DATASET = trento_dataset(data_dir=data_dir)
    elif dataset == "houston":
        from mcvae.dataset import houston_dataset
        DATASET = houston_dataset(data_dir=data_dir, samples_per_class=SAMPLES_PER_CLASS)

    X_train,y_train = DATASET.train_dataset_labelled.tensors 
    X_test,y_test = DATASET.test_dataset_labelled.tensors 
    X,y = DATASET.full_dataset.tensors 

    metrics_dict = {}

    plt.figure(dpi=500)
    plt.imshow(y.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color))
    plt.axis('off')
    plt.savefig(f"{images_dir}GT.png",bbox_inches='tight', pad_inches=0, dpi=500)

    for file in sorted(os.listdir(os.path.join(outputs_dir))):
        if os.path.splitext(file)[-1].lower()=='.npy':
            model_name = file[:-4]
            print(f"Procecing {model_name}")
            y_pred_prob = np.load(os.path.join(outputs_dir,file))
            y_pred_max_prob = y_pred_prob.max(1)
            y_pred = y_pred_prob.argmax(1)+1

            for i in range(N_LABELS): y_pred[i] =i # to avoid zero coloring errors

            print(file)

            # Classification Map
            plt.figure(dpi=500)
            plt.imshow(y_pred.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color))
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_PREDICTIONS.png",bbox_inches='tight', pad_inches=0, dpi=500)
        
            # Metrics 
            y_pred = y_pred[y!=0]

            y_labelled = y[y!=0]

            print(y_pred)
            accuracy = 100*accuracy_score(y_labelled, y_pred)
            precision = 100*precision_score(y_labelled, y_pred, average='macro', zero_division = 0)
            recall = 100*recall_score(y_labelled, y_pred, average='macro', zero_division = 0)
            f1 = 100*f1_score(y_labelled, y_pred, average='macro', zero_division = 0)
            kappa = 100*cohen_kappa_score(y_labelled, y_pred)

            metrics_dict[model_name] = [accuracy, precision, recall, f1, kappa]

            print("Accuracy: ", '{0:.2f}'.format(accuracy))
            print("Precision weighted: ", '{0:.2f}'.format(precision))
            print("Recall weighted: ", '{0:.2f}'.format(recall))
            print("F1-score weighted:", '{0:.2f}'.format(f1))
            print("Kappa score:", '{0:.2f}'.format(kappa))

            # print()
            # m_confusion_matrix = confusion_matrix(y_labelled, y_pred, normalize='true')
            # m_confusion_matrix = np.around(m_confusion_matrix.astype('float') / m_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
            # print(generate_latex_confusion_matrix(m_confusion_matrix, caption = model_name))
            
            print("------------------------------")
            plt.close('all')
            
    print(generate_latex_matrix_from_dict(metrics_dict))

    for file in sorted(os.listdir(os.path.join(outputs_dir, 'uncertainties'))):
        print(file)
        uncertainty = np.load(os.path.join(outputs_dir,'uncertainties',file))

        model_name = file[:-4]
        plt.figure(dpi=500)
        plt.imshow(uncertainty.reshape(SHAPE), cmap='coolwarm', 
        # vmin=0, vmax=2.25
        # vmin=0, vmax=6.25
        )
        plt.axis('off')
        # cbar = plt.colorbar(location='top')
        # cbar.ax.tick_params(labelsize =8 )
        plt.savefig(f"{images_dir}{model_name}.png",bbox_inches='tight', pad_inches=0 ,dpi=500)


a = np.array([[0,1]])
pl.figure(figsize=(9, 1.5))
img = pl.imshow(a, cmap="coolwarm")
pl.gca().set_visible(False)
cax = pl.axes([0.1, 0.2, 0.8, 0.6])
pl.colorbar(orientation="horizontal"
, cax=cax
)
pl.savefig(f"{images_dir}colorbar.png")