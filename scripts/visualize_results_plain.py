from email.mime import image
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from scipy import io
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import torch
import tifffile

from mcvae.utils.utility_functions import compute_reject_label

print(os.listdir('outputs/'))

for project_name in os.listdir('outputs/'):
    if project_name == 'trento':
        dataset = 'trento'
        from trento_config import (
            labels,
            color,
            data_dir,
            images_dir,
            outputs_dir,
            PROJECT_NAME,
            N_LABELS,
            SHAPE
        )
    elif project_name == 'houston':
        dataset = 'houston'
        from houston_config import (
            labels,
            color,
            data_dir,
            images_dir,
            outputs_dir,
            PROJECT_NAME,
            N_LABELS,
            SHAPE,
        )
    elif project_name == 'trento_multimodal':
        dataset = 'trento'
        from trento_multimodal_config import (
            labels,
            color,
            data_dir,
            images_dir,
            outputs_dir,
            PROJECT_NAME,
            N_LABELS,
            SHAPE
        )
    elif project_name == 'houston_multimodal':
        dataset = 'houston'
        from houston_multimodal_config import (
            labels,
            color,
            data_dir,
            images_dir,
            outputs_dir,
            PROJECT_NAME,
            N_LABELS,
            SHAPE,
        )

    else:
        continue

    # Accuracies
    print(project_name, dataset)

    with open(f"{outputs_dir}{PROJECT_NAME}.pkl", 'rb') as f:
        data = pickle.load(f)
    print(data[['MODEL_NAME','N_LATENT', 'encoder_type','LR','N_EPOCHS', 'M_ACCURACY',]])
    data_csv = data[['MODEL_NAME','N_LATENT', 'encoder_type','M_ACCURACY',]]
    data_csv.to_csv(f'{outputs_dir}/{PROJECT_NAME}_accuracies.csv')
    data_dict = data.to_dict()        
        
    for i in range(len(data_dict['LR'])):
        encoder_type = data_dict["encoder_type"][i]
        model_name =  data_dict["MODEL_NAME"][i]
        m_confusion_matrix = data_dict["CONFUSION_MATRIX"][i]
        m_confusion_matrix = np.around(m_confusion_matrix.astype('float') / m_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
        train_loss = data_dict["train_LOSS"][i]
        test_loss = data_dict["test_LOSS"][i]

        # Train-Test Loss
        plt.figure(dpi=500)
        plt.plot(train_loss, color="red", label = 'Train Loss')
        plt.plot(test_loss, color="blue", label='Test Loss')
        plt.xlabel("Epochs")
        plt.grid()
        plt.legend()
        plt.savefig(f"{images_dir}{project_name}_{model_name}_{encoder_type}_LOSS.png", pad_inches=0.2, bbox_inches='tight')
        
    if dataset == "trento":
        y = np.array(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"]) # [166,600] 0 to 6
        y_true = y.reshape(-1)
        image_hyper = torch.tensor(tifffile.imread(data_dir+"hyper_Italy.tif")) # [63,166,600]
        x_all = image_hyper.reshape(len(image_hyper),-1)
        x_all = torch.transpose(x_all, 1,0)
        x_min = x_all.min(dim=0)[0] # [65]
        x_max = x_all.max(dim=0)[0] # [65]
        x_all = (x_all- x_min)/(x_max-x_min)
        hyperrgb = np.zeros((166,600,3))
        hyperrgb[:,:,0] = x_all[:,40].reshape(166,600)+0.05
        hyperrgb[:,:,1] = x_all[:,20].reshape(166,600)+0.1
        hyperrgb[:,:,2] = x_all[:,0].reshape(166,600)+0.2
    
    if dataset == "houston":
        y = np.array(tifffile.imread(data_dir+"houston_gt.tif"), dtype = np.int64) # [1202,4768]
        y_true = y.reshape(-1)
        image_hyper = torch.tensor(tifffile.imread(data_dir+"houston_hyper.tif")) # [50,1202,4768]
        x_all = image_hyper.reshape(len(image_hyper),-1)
        x_all = torch.transpose(x_all, 1,0)
        x_min = x_all.min(dim=0)[0] # [57]
        x_max = x_all.max(dim=0)[0] # [57]
        x_all = (x_all- x_min)/(x_max-x_min)
        hyperrgb = np.zeros((1202,4768,3))
        hyperrgb[:,:,0] = x_all[:,40].reshape(1202,4768)+0.05
        hyperrgb[:,:,1] = x_all[:,20].reshape(1202,4768)+0.1
        hyperrgb[:,:,2] = x_all[:,0].reshape(1202,4768)+0.2


    for file in os.listdir(os.path.join(outputs_dir)):
        if os.path.splitext(file)[-1].lower()=='.npy':
            model_name = file[:-4]

            y_pred_prob = np.load(os.path.join(outputs_dir,file))
            y_pred_max_prob = y_pred_prob.max(1)
            y_pred = y_pred_prob.argmax(1)+1
            y_pred_reject = compute_reject_label(y_pred_prob, threshold=0.5)
            

            plt.figure(dpi=500)
            plt.imshow(y_pred.reshape(SHAPE), interpolation='nearest', cmap = colors.ListedColormap(color[1:]))
            plt.axis('off')
            plt.savefig(f"{images_dir}{model_name}_PREDICTIONS.png",bbox_inches='tight', pad_inches=0, dpi=500)

            plt.figure(dpi=500)
            plt.imshow((y_pred_max_prob*(1-y_pred_max_prob)).reshape(SHAPE))
            plt.axis('off')
            cbar = plt.colorbar(location='top')
            cbar.ax.tick_params(labelsize =8 )
            plt.savefig(f"{images_dir}{model_name}_UNCERTAINTY.png",bbox_inches='tight', pad_inches=0.1 ,dpi=500)

            # Total Confusion matrix
            m_confusion_matrix = confusion_matrix(y_true, y_pred, normalize='true')
            m_confusion_matrix = m_confusion_matrix[1:,1:]
            m_confusion_matrix = np.around(m_confusion_matrix.astype('float') / m_confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

            plt.figure(dpi=500)
            plt.matshow(m_confusion_matrix, cmap="YlGn")
            plt.xlabel("True Labels")
            plt.xticks(np.arange(0,N_LABELS,1), labels[1:])
            plt.ylabel("Predicted Labels")
            plt.yticks(np.arange(0,N_LABELS,1), labels[1:])
            for k in range (len(m_confusion_matrix)):
                for l in range(len(m_confusion_matrix[k])):
                    plt.text(k,l,str(m_confusion_matrix[k][l]), va='center', ha='center', fontsize='xx-small')
            plt.savefig(f"{images_dir}{model_name}_CONFUSION_MATRIX.png",bbox_inches='tight', pad_inches=0.2, dpi=500)
            np.savetxt(f"{outputs_dir}{model_name}_total_confusion_matrix.csv", m_confusion_matrix.astype(int), delimiter=',')





        

        


