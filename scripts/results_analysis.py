"""
This Python script creates classification and uncertainty maps for the 
different projects in the /outputs folder. For Trento and Houston it 
also some zoom in areas of interest.

Usage:
  python3 scripts/results_analysis.py 

"""


from email.mime import image
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from scipy import io
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score
import os
import numpy as np
import torch
import tifffile
import csv
import pylab as pl
import pandas as pd
import seaborn as sns
from PIL import Image

from mcvae.utils import generate_latex_matrix_from_dict, generate_latex_confusion_matrix, crop_npy

print(os.listdir('outputs/'))

for project_name in os.listdir('outputs/'):
    if project_name == 'trento':
        dataset = 'trento'
        from trento_config import *
    elif project_name == 'trento_patch':
        dataset = 'trento'
        from trento_patch_config import *
    elif project_name == 'trento_multimodal':
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
        continue
        from mcvae.dataset import trento_dataset
        DATASET = trento_dataset(data_dir=data_dir)
        (r,g,b) = (31,17,8)
    elif dataset == "houston":
        from mcvae.dataset import houston_dataset
        DATASET = houston_dataset(data_dir=data_dir, samples_per_class=SAMPLES_PER_CLASS)
        # (r,g,b) = (16,13,6) #real
        # (r,g,b) = (40,25,5) # orange
        # (r,g,b) = (20,30,10) # green
        (r,g,b) = (25,30,15) # good contrast
        # (r,g,b) = (50,51,52) # multispectral LiDAR

    X_train,y_train = DATASET.train_dataset_labelled.tensors 
    X_test,y_test = DATASET.test_dataset_labelled.tensors 
    X,y = DATASET.full_dataset.tensors 

    metrics_dict = {}

    plt.figure(dpi=500)



    X_rgb = torch.stack( (X[:,r].reshape(SHAPE), X[:,g].reshape(SHAPE), X[:,b].reshape(SHAPE)), axis=-1) 
    X_rgb = X_rgb.numpy()
    plt.imshow(X_rgb, interpolation='nearest', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(f"{images_dir}{dataset}_RGB.png",bbox_inches='tight', pad_inches=0, dpi=500)

    if dataset == "houston":
        # zoom in vegetation
        (top, right, size) = (100,1600,1000)
        X_rgb_zoom = crop_npy(X_rgb, top, right, size)
        plt.figure(dpi=500)
        plt.imshow(X_rgb_zoom, interpolation='nearest',vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(f"{images_dir}{dataset}_RGB_veg_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

        # zoom in water 
        (top, right, size) = (0,2500,300)
        X_rgb_zoom = crop_npy(X_rgb, top, right, size)
        plt.figure(dpi=500)
        plt.imshow(X_rgb_zoom, interpolation='nearest',vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(f"{images_dir}{dataset}_RGB_water_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)
        
        # zoom in stadium 
        (top, right, size) = (0,1000,800)
        X_rgb_zoom = crop_npy(X_rgb, top, right, size)
        plt.figure(dpi=500)
        plt.imshow(X_rgb_zoom, interpolation='nearest',vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(f"{images_dir}{dataset}_RGB_stadium_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

        # zoom in roads 
        (top, right, size) = (50,3300,800)
        X_rgb_zoom = crop_npy(X_rgb, top, right, size)
        plt.figure(dpi=500)
        plt.imshow(X_rgb_zoom, interpolation='nearest',vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(f"{images_dir}{dataset}_RGB_roads_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

        # zoom in trains 
        (top, right, size) =  (200,200,800)
        X_rgb_zoom = crop_npy(X_rgb, top, right, size)
        plt.figure(dpi=500)
        plt.imshow(X_rgb_zoom, interpolation='nearest',vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(f"{images_dir}{dataset}_RGB_trains_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

        # zoom in uplots 
        (top, right, size) = (400,300,500)
        X_rgb_zoom = crop_npy(X_rgb, top, right, size)
        plt.figure(dpi=500)
        plt.imshow(X_rgb_zoom, interpolation='nearest',vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(f"{images_dir}{dataset}_RGB_uplots_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

        # zoom in cars 
        (top, right, size) = (700,1450,500)
        X_rgb_zoom = crop_npy(X_rgb, top, right, size)
        plt.figure(dpi=500)
        plt.imshow(X_rgb_zoom, interpolation='nearest',vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(f"{images_dir}{dataset}_RGB_cars_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)


    plt.figure(dpi=500)
    plt.imshow(y.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color))
    plt.axis('off')
    plt.savefig(f"{images_dir}{dataset}_GT.png",bbox_inches='tight', pad_inches=0, dpi=500)

    for file in sorted(os.listdir(os.path.join(outputs_dir))):
        if os.path.splitext(file)[-1].lower()=='.npy':
            model_name = file[:-4]
            print(f"Procecing {model_name}")
            y_pred_prob = np.load(os.path.join(outputs_dir,file))
            print(y_pred_prob.shape)
            y_pred_max_prob = y_pred_prob.max(1)
            y_pred = y_pred_prob.argmax(1)+1

            for i in range(N_LABELS): y_pred[i] =i # to avoid zero coloring errors

            print(file)

            # Classification Map
            y_pred_all = y_pred.reshape(SHAPE)
            for i in range(N_LABELS): y_pred_all[-4:,i] =i # to avoid zero coloring errors

            plt.figure(dpi=500)
            plt.imshow(y_pred_all, interpolation='nearest', cmap = colors.ListedColormap(color))
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_PREDICTIONS.png",bbox_inches='tight', pad_inches=0, dpi=500)

            if dataset == "houston":
                # zoom in vegetation
                (top, right, size) = (100,1600,1000)
                y_pred_zoom = crop_npy(y_pred.reshape(SHAPE), top, right, size)
                for i in range(N_LABELS): y_pred_zoom[-1,i] =i # to avoid zero coloring errors
                plt.figure(dpi=500)
                plt.imshow(y_pred_zoom, interpolation='nearest', cmap = colors.ListedColormap(color))
                plt.axis('off')
                plt.savefig(f"{images_dir}{model_name}_PREDICTIONS_veg_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)
                
                # zoom in water
                (top, right, size) = (0,2500,300)
                y_pred_zoom = crop_npy(y_pred.reshape(SHAPE), top, right, size)
                for i in range(N_LABELS): y_pred_zoom[-1:,i] =i # to avoid zero coloring errors
                plt.figure(dpi=500)
                plt.imshow(y_pred_zoom, interpolation='nearest', cmap = colors.ListedColormap(color))
                plt.axis('off')
                plt.savefig(f"{images_dir}{model_name}_PREDICTIONS_water_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)
                break

                # zoom in stadium
                (top, right, size) = (0,1000,800)
                y_pred_zoom = crop_npy(y_pred.reshape(SHAPE), top, right, size)
                for i in range(N_LABELS): y_pred_zoom[-1,i] =i # to avoid zero coloring errors
                plt.figure(dpi=500)
                plt.imshow(y_pred_zoom, interpolation='nearest', cmap = colors.ListedColormap(color))
                plt.axis('off')
                plt.savefig(f"{images_dir}{model_name}_PREDICTIONS_stadium_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)
    
                # zoom in roads
                (top, right, size) = (50,3300,800)
                y_pred_zoom = crop_npy(y_pred.reshape(SHAPE), top, right, size)
                for i in range(N_LABELS): y_pred_zoom[-1,i] =i # to avoid zero coloring errors
                plt.figure(dpi=500)
                plt.imshow(y_pred_zoom, interpolation='nearest', cmap = colors.ListedColormap(color))
                plt.axis('off')
                plt.savefig(f"{images_dir}{model_name}_PREDICTIONS_roads_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

                # zoom in trains
                (top, right, size) =  (200,200,800)
                y_pred_zoom = crop_npy(y_pred.reshape(SHAPE), top, right, size)
                for i in range(N_LABELS): y_pred_zoom[-1,i] =i # to avoid zero coloring errors
                plt.figure(dpi=500)
                plt.imshow(y_pred_zoom, interpolation='nearest', cmap = colors.ListedColormap(color))
                plt.axis('off')
                plt.savefig(f"{images_dir}{model_name}_PREDICTIONS_trains_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

                # zoom in uplots
                (top, right, size) = (400,300,500)  
                y_pred_zoom = crop_npy(y_pred.reshape(SHAPE), top, right, size)
                for i in range(N_LABELS): y_pred_zoom[0,i] =i # to avoid zero coloring errors
                plt.figure(dpi=500)
                plt.imshow(y_pred_zoom, interpolation='nearest', cmap = colors.ListedColormap(color))
                plt.axis('off')
                plt.savefig(f"{images_dir}{model_name}_PREDICTIONS_uplots_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

                # zoom in cars
                (top, right, size) = (700,1450,500)
                y_pred_zoom = crop_npy(y_pred.reshape(SHAPE), top, right, size)
                print(np.unique(y_pred_zoom))
                for i in range(N_LABELS): y_pred_zoom[0,i] =i # to avoid zero coloring errors
                print(np.unique(y_pred_zoom))
                print(color)
                plt.figure(dpi=500)
                plt.imshow(y_pred_zoom, interpolation='nearest', cmap = colors.ListedColormap(color))
                plt.axis('off')
                plt.savefig(f"{images_dir}{model_name}_PREDICTIONS_cars_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)


            # Metrics 
            y_pred = y_pred[y!=0]

            y_labelled = y[y!=0]

            print(classification_report(y_labelled, y_pred))

            # print(y_pred)
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

            print()
            m_confusion_matrix = confusion_matrix(y_labelled, y_pred, normalize='true')
            m_confusion_matrix = np.around(m_confusion_matrix.astype('float') / m_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
            print(generate_latex_confusion_matrix(m_confusion_matrix, caption = model_name))
            
            print("------------------------------")
            plt.close('all')
    
    print(generate_latex_matrix_from_dict(metrics_dict))


    models_names = list(metrics_dict.keys())
    metrics_values = {
        'Accucaracy':[ round(metrics_dict[k][0],2) for k in metrics_dict.keys()],
        'Precision': [ round(metrics_dict[k][1],2) for k in metrics_dict.keys()],
        'Recall': [ round(metrics_dict[k][2],2) for k in metrics_dict.keys()],
        'F1': [ round(metrics_dict[k][3],2) for k in metrics_dict.keys()],
    }

    print(models_names)
    print(metrics_values)

    for file in sorted(os.listdir(os.path.join(outputs_dir, 'uncertainties'))):
        print(file)
        uncertainty = np.load(os.path.join(outputs_dir,'uncertainties',file))

        model_name = file[:-4]
        
        plt.figure(dpi=500)
        plt.imshow(uncertainty.reshape(SHAPE), cmap='turbo', 
        vmin=0, vmax=1
        )
        plt.axis('off')
        plt.savefig(f"{images_dir}{model_name}.png",bbox_inches='tight', pad_inches=0 ,dpi=500)

        if dataset == "houston":
            # zoom in vegetation
            (top, right, size) = (100,1600,1000)
            uncertainty_zoom = crop_npy(uncertainty.reshape(SHAPE), top, right, size)
            plt.figure(dpi=500)
            plt.imshow(uncertainty_zoom, interpolation='nearest', cmap = 'turbo', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_veg_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

            # zoom in water
            (top, right, size) = (0,2500,300)
            uncertainty_zoom = crop_npy(uncertainty.reshape(SHAPE), top, right, size)
            plt.figure(dpi=500)
            plt.imshow(uncertainty_zoom, interpolation='nearest', cmap = 'turbo', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_water_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

            # zoom in stadium
            (top, right, size) = (0,1000,800)
            uncertainty_zoom = crop_npy(uncertainty.reshape(SHAPE), top, right, size)
            plt.figure(dpi=500)
            plt.imshow(uncertainty_zoom, interpolation='nearest', cmap = 'turbo', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_stadium_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

            # zoom in roads
            (top, right, size) = (50,3300,800)
            uncertainty_zoom = crop_npy(uncertainty.reshape(SHAPE), top, right, size)
            plt.figure(dpi=500)
            plt.imshow(uncertainty_zoom, interpolation='nearest', cmap = 'turbo', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_roads_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

            # zoom in trains
            (top, right, size) = (200,200,800)
            uncertainty_zoom = crop_npy(uncertainty.reshape(SHAPE), top, right, size)
            plt.figure(dpi=500)
            plt.imshow(uncertainty_zoom, interpolation='nearest', cmap = 'turbo', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_trains_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

            # zoom in uplots
            (top, right, size) = (400,500,500)
            uncertainty_zoom = crop_npy(uncertainty.reshape(SHAPE), top, right, size)
            plt.figure(dpi=500)
            plt.imshow(uncertainty_zoom, interpolation='nearest', cmap = 'turbo', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_uplots_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)

            # zoom in cars
            (top, right, size) = (700,1450,500)
            uncertainty_zoom = crop_npy(uncertainty.reshape(SHAPE), top, right, size)
            plt.figure(dpi=500)
            plt.imshow(uncertainty_zoom, interpolation='nearest', cmap = 'turbo', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_cars_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)


a = np.array([[0,1]])
pl.figure(figsize=(9, 1.5))
img = pl.imshow(a, cmap="turbo")
pl.gca().set_visible(False)
cax = pl.axes([0.1, 0.2, 0.8, 0.6])
pl.colorbar(orientation="horizontal"
, cax=cax
)
pl.savefig(f"{images_dir}colorbar.png")