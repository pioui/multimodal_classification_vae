from email.mime import image
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from scipy import io
from sklearn.metrics import confusion_matrix
import os
import numpy as np
from trento_utils import compute_reject_label
import torch
import tifffile
from trento_config import (
    labels,
    color,
    data_dir,
    images_dir,
    outputs_dir,
)

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

with open(f"{outputs_dir}trento.pkl", 'rb') as f:
    data = pickle.load(f)

print(data[['MODEL_NAME','N_LATENT', 'encoder_type','M_ACCURACY',]])
data_csv = data[['MODEL_NAME','N_LATENT', 'encoder_type','M_ACCURACY',]]
data_csv.to_csv(f'{outputs_dir}/accuracies.csv')
data_dict = data.to_dict()

for i in range(len(data_dict['LR'])):
    encoder_type = data_dict["encoder_type"][i]
    model_name =  data_dict["MODEL_NAME"][i]
    m_confusion_matrix = data_dict["CONFUSION_MATRIX"][i]
    train_loss = data_dict["train_LOSS"][i]
    test_loss = data_dict["test_LOSS"][i]

    plt.figure()
    plt.plot(train_loss, color="red", label = 'Train Loss')
    plt.plot(test_loss, color="blue", label='Test Loss')
    plt.title("Train Test Loss")
    plt.xlabel("Epochs")
    plt.grid()
    plt.legend()
    plt.savefig(f"{images_dir}{model_name}_{encoder_type}_test_loss.png")

    plt.figure()
    plt.matshow(m_confusion_matrix, cmap="YlGn")
    plt.xlabel("True Labels")
    plt.xticks(np.arange(0,6,1), labels[1:])
    plt.ylabel("Predicted Labels")
    plt.yticks(np.arange(0,6,1), labels[1:])

    for k in range (len(m_confusion_matrix)):
        for l in range(len(m_confusion_matrix[k])):
            plt.text(k,l,str(m_confusion_matrix[k][l]), va='center', ha='center')
    plt.title("Test Confusion Matrix")

    plt.savefig(f"{images_dir}{model_name}_{encoder_type}_test_confusion_matrix.png", bbox_inches='tight')

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


for subdir, dir, files in os.walk(outputs_dir):
    for file in files:
        if os.path.splitext(file)[-1].lower()=='.npy':
            model_name = file[:-4]

            y_pred_prob = np.load(os.path.join(outputs_dir,file))
            y_pred = y_pred_prob.argmax(1)+1
            y_pred_reject = compute_reject_label(y_pred_prob, threshold=0.7)

            plt.figure(dpi=1000)
            plt.subplot(411)
            plt.imshow(hyperrgb)
            plt.axis('off')
            plt.title("HSI", fontsize=6)
            plt.subplot(412)
            plt.imshow(y_true.reshape(166,600), interpolation='nearest', cmap = colors.ListedColormap(color))
            plt.axis('off')
            plt.title("Ground Truth", fontsize=6)
            plt.subplot(413)
            plt.imshow(y_pred.reshape(166,600), interpolation='nearest', cmap = colors.ListedColormap(color[1:]))
            plt.axis('off')
            plt.title("Predictions", fontsize=6)
            plt.subplot(414)
            plt.imshow(y_pred_reject.reshape(166,600), interpolation='nearest', cmap = colors.ListedColormap(color))
            plt.axis('off')
            plt.title("Predictions with Uknown Class", fontsize=6)

            handles = []
            for c,l in zip(color, labels):
                handles.append(mpatches.Patch(color=c, label=l))
            plt.legend(handles=handles, loc='lower center', prop={'size':10}, bbox_to_anchor=(0.5,-1), ncol=3, borderaxespad=0.)
            plt.savefig(f"{images_dir}{model_name}_classification_matrix.png",bbox_inches='tight')

            m_confusion_matrix = confusion_matrix(y_true, y_pred)
            m_confusion_matrix = m_confusion_matrix[1:,1:]
            plt.figure()
            plt.matshow(m_confusion_matrix, cmap="YlGn")
            plt.xlabel("True Labels")
            plt.xticks(np.arange(0,6,1), labels[1:])
            plt.ylabel("Predicted Labels")
            plt.yticks(np.arange(0,6,1), labels[1:])
            for k in range (len(m_confusion_matrix)):
                for l in range(len(m_confusion_matrix[k])):
                    plt.text(k,l,str(m_confusion_matrix[k][l]), va='center', ha='center')
            plt.title("Total Confusion Matrix")
            plt.savefig(f"{images_dir}{model_name}_total_confusion_matrix.png",bbox_inches='tight')

            m_confusion_matrix = confusion_matrix(y_true, y_pred_reject)
            m_confusion_matrix = m_confusion_matrix[1:,1:]
            plt.figure()
            plt.matshow(m_confusion_matrix, cmap="YlGn")
            plt.xlabel("True Labels")
            plt.xticks(np.arange(0,6,1), labels[1:])
            plt.ylabel("Predicted Labels")
            plt.yticks(np.arange(0,6,1), labels[1:])
            for k in range (len(m_confusion_matrix)):
                for l in range(len(m_confusion_matrix[k])):
                    plt.text(k,l,str(m_confusion_matrix[k][l]), va='center', ha='center')
            plt.title("Total Confusion Matrix")
            plt.savefig(f"{images_dir}{model_name}_reject_total_confusion_matrix.png",bbox_inches='tight')


