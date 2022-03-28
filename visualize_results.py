import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from scipy import io
from sklearn.metrics import confusion_matrix
import os
import numpy as np
from trento_utils import compute_reject_label

from trento_config import (
    labels,
    color,
    data_dir,
    outputs_dir
)

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

with open(f"{outputs_dir}trento.pkl", 'rb') as f:
    data = pickle.load(f)

print(data[['MODEL_NAME','N_LATENT', 'encoder_type','M_ACCURACY',]])

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
    plt.savefig(f"{outputs_dir}{model_name}_{encoder_type}_test_loss.png")

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

    plt.savefig(f"{outputs_dir}{model_name}_{encoder_type}_test_confusion_matrix.png", bbox_inches='tight')


y = np.array(io.loadmat(data_dir+"TNsecSUBS_Test.mat")["TNsecSUBS_Test"]) # [166,600] 0 to 6
y_true = y.reshape(-1)

for subdir, dir, files in os.walk(outputs_dir):
    for file in files:
        if os.path.splitext(file)[-1].lower()=='.npy':
            model_name = file[:-4]

            y_pred_prob = np.load(os.path.join(outputs_dir,file))
            y_pred = y_pred_prob.argmax(1)+1
            
            plt.figure()
            plt.subplot(211)
            plt.imshow(y_pred.reshape(166,600), interpolation='nearest', cmap = colors.ListedColormap(color[1:]))
            plt.axis('off')
            plt.title("Predictions")
            plt.subplot(212)
            plt.imshow(y_true.reshape(166,600), interpolation='nearest', cmap = colors.ListedColormap(color))
            plt.axis('off')
            plt.title("Ground Truth")
            handles = []
            for c,l in zip(color, labels):
                handles.append(mpatches.Patch(color=c, label=l))
            plt.legend(handles=handles, loc='lower center', prop={'size':10}, bbox_to_anchor=(0.5,-0.55), ncol=4, borderaxespad=0.)
            plt.savefig(f"{outputs_dir}{model_name}_classification_matrix.png",bbox_inches='tight')

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
            plt.savefig(f"{outputs_dir}{model_name}_total_confusion_matrix.png",bbox_inches='tight')

            y_pred = compute_reject_label(y_pred_prob, threshold=0.5)

            plt.figure()
            plt.subplot(211)
            plt.imshow(y_pred.reshape(166,600), interpolation='nearest', cmap = colors.ListedColormap(color))
            plt.axis('off')
            plt.title("Predictions")
            plt.subplot(212)
            plt.imshow(y_true.reshape(166,600), interpolation='nearest', cmap = colors.ListedColormap(color))
            plt.axis('off')
            plt.title("Ground Truth")
            handles = []
            for c,l in zip(color, labels):
                handles.append(mpatches.Patch(color=c, label=l))
            plt.legend(handles=handles, loc='lower center', prop={'size':10}, bbox_to_anchor=(0.5,-0.55), ncol=4, borderaxespad=0.)
            plt.savefig(f"{outputs_dir}{model_name}_reject_classification_matrix.png",bbox_inches='tight')

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
            plt.savefig(f"{outputs_dir}{model_name}_reject_total_confusion_matrix.png",bbox_inches='tight')


