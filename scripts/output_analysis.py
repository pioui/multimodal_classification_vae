from email.mime import image
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from scipy import io
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import os
import numpy as np
import torch
import tifffile
import csv

from mcvae.utils.utility_functions import compute_reject_label, centroid, entropy, variance, variance_heterophil

print(os.listdir('outputs/'))

for project_name in os.listdir('outputs/'):
    if project_name == 'trento':
        # continue
        print(project_name)
        dataset = 'trento'
        from trento_config import *
    elif project_name == 'trento_patch':
        # continue
        dataset = 'trento'
        from trento_patch_config import *
    elif project_name == 'houston':
        # continue
        dataset = 'houston'
        from houston_config import *
    elif project_name == 'houston_patch':
        # continue
        dataset = 'houston'
        from houston_patch_config import *
    elif project_name == 'trento_multimodal':
        # continue
        dataset = 'trento'
        from trento_multimodal_config import *
    elif project_name == 'houston_multimodal':
        # continue
        dataset = 'houston'
        from houston_multimodal_config import *
    else:
        continue

    # Accuracies
    print(project_name, dataset)

    if os.path.exists(f"{outputs_dir}{PROJECT_NAME}.pkl"):

        with open(f"{outputs_dir}{PROJECT_NAME}.pkl", 'rb') as f:
            data = pickle.load(f)
        print(data[['MODEL_NAME','N_LATENT', 'encoder_type','LR','N_EPOCHS', 'M_ACCURACY']])
        data_csv = data[['MODEL_NAME','N_LATENT', 'encoder_type','M_ACCURACY']]
        data_csv.to_csv(f'{outputs_dir}/{PROJECT_NAME}_test_accuracies.csv')
        data_dict = data.to_dict()        
            
        for i in range(len(data_dict['LR'])):
            encoder_type = data_dict["encoder_type"][i]
            model_name =  data_dict["MODEL_NAME"][i]
            m_confusion_matrix = data_dict["CONFUSION_MATRIX"][i][1:,1:]
            m_confusion_matrix = np.around(m_confusion_matrix.astype('float') / m_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
            train_loss = data_dict["train_LOSS"][i]
            test_loss = data_dict["test_LOSS"][i]

            # Train-Test Loss
            plt.figure(dpi=500)
            plt.plot(train_loss, color="red", label = 'Train Loss')
            plt.plot(test_loss, color="blue", label='Test Loss')
            plt.xlabel("Epochs")
            # plt.ylim(-1.2,2) #Trento
            plt.ylim(-0.35,0.35) # Houston
            plt.grid()
            plt.legend()
            plt.savefig(f"{images_dir}{project_name}_{model_name}_{encoder_type}_LOSS.png", pad_inches=0.2, bbox_inches='tight')
            
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
            plt.savefig(f"{images_dir}{project_name}_{model_name}_{encoder_type}_test_CONFUSION_MATRIX.png",bbox_inches='tight', pad_inches=0.2, dpi=500)
            np.savetxt(f"{outputs_dir}{project_name}_{model_name}_{encoder_type}_test_CONFUSION_MATRIX.csv", m_confusion_matrix.astype(int), delimiter=',')
    else:
        print("No .pkl file")


    if dataset == "trento":
        y = np.array(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"]) # [166,600] 0 to 6
        y_true = y.reshape(-1)
    
    if dataset == "houston":
        y = np.array(tifffile.imread(data_dir+"houston_gt.tif"), dtype = np.int64) # [1202,4768]
        y_true = y.reshape(-1)


    acc_dict = []
    for file in os.listdir(os.path.join(outputs_dir)):
        print(file)
        if os.path.splitext(file)[-1].lower()=='.npy':
            model_name = file[:-4]

            y_pred_prob = np.load(os.path.join(outputs_dir,file))
            print(y_pred_prob.shape)
            y_pred_max_prob = y_pred_prob.max(1)
            y_pred = y_pred_prob.argmax(1)+1
            y_pred_reject = compute_reject_label(y_pred_prob, threshold=0.5)
            

            plt.figure(dpi=500)
            plt.imshow(y_pred.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color[1:]))
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_PREDICTIONS.png",bbox_inches='tight', pad_inches=0, dpi=500)

            # plt.figure(dpi=500)
            # plt.imshow((1-y_pred_max_prob).reshape(SHAPE), cmap='coolwarm', 
            # vmin=0, vmax=1
            # )
            # plt.axis('off')
            # cbar = plt.colorbar(location='top')
            # cbar.ax.tick_params(labelsize =8 )
            # plt.savefig(f"{images_dir}{model_name}_UNCERTAINTY.png",bbox_inches='tight', pad_inches=0.1 ,dpi=500)
            
            # plt.figure(dpi=500)
            # plt.imshow(centroid(y_pred_prob).reshape(SHAPE), cmap='coolwarm', 
            # vmin=0, vmax=1
            # )
            # plt.axis('off')
            # cbar = plt.colorbar(location='top')
            # cbar.ax.tick_params(labelsize =8 )
            # plt.savefig(f"{images_dir}{model_name}_CENTROID.png",bbox_inches='tight', pad_inches=0.1 ,dpi=500)

            # plt.figure(dpi=500)
            # plt.imshow(variance(y_pred_prob).reshape(SHAPE), cmap='coolwarm', 
            # # vmin=0, vmax=1
            # )
            # plt.axis('off')
            # cbar = plt.colorbar(location='top')
            # cbar.ax.tick_params(labelsize =8 )
            # plt.savefig(f"{images_dir}{model_name}_VARIANCE.png",bbox_inches='tight', pad_inches=0.1 ,dpi=500)

            # var = variance_heterophil(p = y_pred_prob, w = heterophil_matrix )
            # plt.figure(dpi=500)
            # plt.imshow(var.reshape(SHAPE), cmap='coolwarm', 
            # # vmin=0, vmax=2.25
            # vmin=0, vmax=6.25
            # )
            plt.axis('off')
            cbar = plt.colorbar(location='top')
            cbar.ax.tick_params(labelsize =8 )
            plt.savefig(f"{images_dir}{model_name}_VARIANCE_HETERO.png",bbox_inches='tight', pad_inches=0.1 ,dpi=500)

            # plt.figure(dpi=500)
            # plt.imshow(entropy(y_pred_prob).reshape(SHAPE), cmap='coolwarm', 
            # # vmin=0, vmax=1
            # )
            # plt.axis('off')
            # cbar = plt.colorbar(location='top')
            # cbar.ax.tick_params(labelsize =8 )
            # plt.savefig(f"{images_dir}{model_name}_ENTROPY.png",bbox_inches='tight', pad_inches=0.1 ,dpi=500)

            # Total Confusion matrix
            m_confusion_matrix = confusion_matrix(y_true, y_pred, normalize='true')
            m_confusion_matrix = m_confusion_matrix[1:,1:]
            m_confusion_matrix = np.around(m_confusion_matrix.astype('float') / m_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

            plt.figure(dpi=500)
            plt.matshow(m_confusion_matrix, cmap="coolwarm")

            plt.xlabel("True Labels")
            plt.xticks(np.arange(0,N_LABELS,1), range(1,len(labels)))
            plt.ylabel("Predicted Labels")
            plt.yticks(np.arange(0,N_LABELS,1), range(1,len(labels)))
            for k in range (len(m_confusion_matrix)):
                for l in range(len(m_confusion_matrix[k])):
                    # plt.text(k,l,str(m_confusion_matrix[k][l]), va='center', ha='center', fontsize='small') # trento
                    plt.text(k,l,str(m_confusion_matrix[k][l]), va='center', ha='center', fontsize='xx-small') # houston
            plt.savefig(f"{images_dir}{model_name}_total_CONFUSION_MATRIX.png",bbox_inches='tight', pad_inches=0.2, dpi=500)
            np.savetxt(f"{outputs_dir}{model_name}_total_CONFUSION_MATRIX.csv", m_confusion_matrix.astype(int), delimiter=',')

            # Total Accuracy
            indeces = (y_true!=0)
            m_accuracy = accuracy_score(y_true[indeces],  y_pred[indeces])
            b_accuracy = balanced_accuracy_score(y_true[indeces],  y_pred[indeces])

            # Unknown label

            y_pred[y_pred<0.5]=0
            indeces = (y_true!=0)
            um_accuracy = accuracy_score(y_true[indeces],  y_pred[indeces])
            ub_accuracy = balanced_accuracy_score(y_true[indeces],  y_pred[indeces])
            acc_dict.append({
                'model_name':model_name, 
                'accuracy':m_accuracy, 
                'balanced accuracy': b_accuracy,
                'unkown label accuracy': um_accuracy, 
                'uknown label balanced accuracy': ub_accuracy,
                })

            # # Uncertain label
            # print(y_pred_prob[var>2])
            # print(y_pred[var>2])
            # print(y_pred_prob.shape)
            # two_classes = (-y_pred_prob).argsort(1)[:,:2]
            # import sys
            # np.set_printoptions(threshold=sys.maxsize)

            # print(two_classes.shape)
            # print(two_classes[var>2])

    with open(f'{outputs_dir}/{PROJECT_NAME}_total_accuracies.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['model_name', 'accuracy', 'balanced accuracy','unkown label accuracy','uknown label balanced accuracy' ])
        writer.writeheader()
        writer.writerows(acc_dict)


        


