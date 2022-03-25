import pickle


with open('trento-relaxed_nparticules_30.pkl', 'rb') as f:
    data = pickle.load(f)

print(data[['encoder_type','LOSS_GEN', 'LOSS_WVAR','MODEL_NAME','train_M_ACCURACY','M_ACCURACY','train_M_ACCURACY_IS','M_ACCURACY_IS' ]])