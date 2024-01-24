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
import pandas as pd
import seaborn as sns
from PIL import Image

from mcvae.utils import generate_latex_matrix_from_dict, generate_latex_confusion_matrix, crop_npy

print(os.listdir('outputs/'))

# Function to calculate accuracy for a single model
def calculate_accuracy(ground_truth, predictions):
    correct = np.sum(ground_truth == predictions)
    total = len(ground_truth)
    accuracy = correct / total
    return accuracy

# Function to read ground truth and predictions, and calculate metrics
def process_model(model_name, ground_truth_file, predictions_file):
    ground_truth = np.loadtxt(ground_truth_file, dtype=int)
    predictions = np.loadtxt(predictions_file, dtype=int)

    # Calculate accuracy for each class
    unique_classes = np.unique(ground_truth)
    class_accuracies = []
    for class_label in unique_classes:
        class_mask = (ground_truth == class_label)
        class_accuracy = calculate_accuracy(ground_truth[class_mask], predictions[class_mask])
        class_accuracies.append(class_accuracy)

    # Calculate average accuracy
    average_accuracy = np.mean(class_accuracies)

    # Calculate overall accuracy
    overall_accuracy = calculate_accuracy(ground_truth, predictions)

    return [model_name, *class_accuracies, average_accuracy, overall_accuracy]




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
        from mcvae.dataset import trento_dataset
        DATASET = trento_dataset(data_dir=data_dir)
        (r,g,b) = (31,17,8)
    elif dataset == "houston":
        from mcvae.dataset import houston_dataset
        DATASET = houston_dataset(data_dir=data_dir, samples_per_class=SAMPLES_PER_CLASS)
        # (r,g,b) = (16,13,6) #real
        (r,g,b) = (40,25,5) # orange
        (r,g,b) = (20,30,10) # green
        (r,g,b) = (25,30,15) # good contrast

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

        # # zoom in cars 
        # (top, right, size) = (700,1450,500)
        # X_rgb_zoom = crop_npy(X_rgb, top, right, size)
        # plt.figure(dpi=500)
        # plt.imshow(X_rgb_zoom, interpolation='nearest',vmin=0, vmax=1)
        # plt.axis('off')
        # plt.savefig(f"{images_dir}{dataset}_RGB_cars_zoom.png",bbox_inches='tight', pad_inches=0, dpi=500)


    plt.figure(dpi=500)
    plt.imshow(y.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color))
    plt.axis('off')
    plt.savefig(f"{images_dir}{dataset}_GT.png",bbox_inches='tight', pad_inches=0, dpi=500)
    model_names = []
    accuracies = {}

    for file in sorted(os.listdir(os.path.join(outputs_dir))):
        if os.path.splitext(file)[-1].lower()=='.npy':
            model_name = file[:-4]

            print(f"Procecing {model_name}")
            y_pred_prob = np.load(os.path.join(outputs_dir,file))
            y_pred_max_prob = y_pred_prob.max(1)
            y_pred = y_pred_prob.argmax(1)+1

            for i in range(N_LABELS): y_pred[i] =i # to avoid zero coloring errors


            # Classification Map
            y_pred_all = y_pred.reshape(SHAPE)
            for i in range(N_LABELS): y_pred_all[-4:,i] =i # to avoid zero coloring errors


            # Metrics 
            y_pred = y_pred[y!=0]

            y_labelled = y[y!=0]

            ground_truth = y_labelled

            class_labels = np.unique(ground_truth)
            class_accuracies = []
            for label in class_labels:
                mask = ground_truth == label
                class_accuracy = accuracy_score(ground_truth[mask], y_pred[mask])
                class_accuracies.append(class_accuracy)
        else:
            continue

    # Calculate average and overall accuracy for all models
    average_accuracies = {model: np.mean(class_accuracies) for model, class_accuracies in accuracies.items()}
    overall_accuracies = {model: accuracy_score(ground_truth, predictions) for model, predictions in zip(model_names, [model1_predictions, model2_predictions, model3_predictions, model4_predictions])}

    # Create a table in LaTeX format
    table_headers = ["Class"] + model_names
    table_data = []

    for label in class_labels:
        row = [label]
        for model in model_names:
            row.append(f"{accuracies[model][label]:.2%}")
        table_data.append(row)

    average_row = ["Average"]
    for model in model_names:
        average_row.append(f"{average_accuracies[model]:.2%}")

    overall_row = ["Overall"]
    for model in model_names:
        overall_row.append(f"{overall_accuracies[model]:.2%}")

    table_data.append(average_row)
    table_data.append(overall_row)

    latex_table = tabulate(table_data, headers=table_headers, tablefmt="latex_booktabs")

    # Print the LaTeX table
    print(latex_table)

            



