import pickle
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import os
import numpy as np

if not os.path.exists("outputs/"):
    os.makedirs("outputs/")

with open('trento-relaxed_nparticules_30.pkl', 'rb') as f:
    data = pickle.load(f)

print(data[['MODEL_NAME', 'encoder_type','M_ACCURACY',]])

data_dict = data.to_dict()

for i in range(len(data_dict['LR'])):
    encoder_type = data_dict["encoder_type"][i]
    model_name =  data_dict["MODEL_NAME"][i]
    confusion_matrix = data_dict["CONFUSION_MATRIX"][i]
    train_loss = data_dict["train_LOSS"][i]
    test_loss = data_dict["test_LOSS"][i]

    plt.figure()
    plt.plot(train_loss, color="red")
    plt.plot(test_loss, color="blue")
    plt.legend()
    plt.xlabel("Epochs")
    plt.title("Test and Train Loss")
    plt.grid()
    plt.savefig(f"outputs/{model_name}_{encoder_type}_loss.png")

    plt.matshow(confusion_matrix, cmap="YlGn")
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    for k in range (len(confusion_matrix)):
        for l in range(len(confusion_matrix[k])):
            plt.text(k,l,str(confusion_matrix[k][l]), va='center', ha='center')
    plt.title("Confusion Matrix")

    plt.savefig(f"outputs/{model_name}_{encoder_type}_confusion_matrix.png")



