import numpy as np 
import pandas as pd 
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras import models
from tensorflow.keras import callbacks
import os

path = '/home/khademg/simulationStudy/scripts/'
os.chdir(path)

from classifiers_dataset_info import *

tf.keras.backend.clear_session()
#================================================================

def read_dataset(path_dir, filename, dataset_dict, dataset_name, scale='normalize'):
    path = path_dir + '/' + filename
    data = pd.read_csv(path)
    n_class = dataset_dict[dataset_name]['nb_classes']
    n_indv_per_class = dataset_dict[dataset_name]['n_indv_per_class']
    n_channels = dataset_dict[dataset_name]['n_channels']
    class_labels = np.array(dataset_dict[dataset_name]['class_labels'])
    ts_l = int(round(len(data.ID)/n_indv_per_class/n_class))  # Length of time series
    sample_size = n_class * n_indv_per_class                       # Sample Size
    X = np.empty((sample_size, ts_l, n_channels))
    y = np.empty((sample_size, n_class), dtype=int)
    ID = ["" for x in range(sample_size)]
    original_class = ["" for x in range(sample_size)]
    if scale == 'normalize':
        mean = np.mean(data.SBP, axis=0)
        std = np.std(data.SBP, axis=0)
        data.SBP -= mean
        data.SBP /= std
    elif scale == 'min_max':
        MIN = np.min(data.SBP, axis=0)
        MAX = np.max(data.SBP, axis=0)
        data.SBP = (2*data.SBP - MAX - MIN)/(MAX - MIN)
        # Floating point inaccuracy!
        data.SBP = np.where(data.SBP >= 1., 1., data.SBP)
        data.SBP = np.where(data.SBP <= -1., -1., data.SBP)
    for i in range(sample_size):
        X[i,:,0] = data.SBP[i*ts_l: (i+1)*ts_l]
        h = np.where(class_labels==data.Class[i*ts_l])
        y[i,] = to_categorical(h[0], n_class)
        ID[i] = data.ID[i*ts_l]
        original_class[i] = data.Class[i*ts_l]
    ID_Class = np.column_stack((ID, original_class))
    return X, y, mean, std

#================================================================
# Load datasets

set_type = 1      # 0: mag_cohort, 1: shape_cohort
method = 1        # 0: ResNet, 1: FCNN


filenames = ['_Simulated_Cohort_2_3000_5.5_2.5.csv', '_Simulated_cohort_5_3000_5_1.5.csv']
dataset_path = ['/home/khademg/simulationStudy/dataMarch2020/data_subset/mag_cohorts/Simulation_Cohort_1086_18323',
             '/home/khademg/simulationStudy/dataMarch2020/data_subset/shape_cohorts/Simulation_Cohort_387_5631091_5']
dataset_name = ['mag_cohorts_NEW', 'shape_cohorts_NEW']
X, Y, mean, std = read_dataset(dataset_path[set_type], filenames[set_type], dataset_dict, dataset_name[set_type])

indexes = np.arange(len(X))
np.random.shuffle(indexes)
n_sample = 30
X_test = X[indexes[:n_sample]]
Y_test = Y[indexes[:n_sample]]

#================================================================
# Load Model - ResNet or FCNN models

classifier = ['ResNet', 'FCNN']
filenames = ['mag_cohorts_NEW_Simulation_Cohort_1086_18323_2_3000_5.5_2.5_',
             'shape_cohorts_NEW_Simulation_Cohort_387_5631091_5_5_3000_5_1.5_']
root_path = '/home/khademg/simulationStudy/scripts/results/NEW_Cohort_Subset/models/'
model_path = root_path + filenames[set_type] + classifier[method] +  '_best_model.hdf5'
model = models.load_model(model_path)

#================================================================

gap_weights = model.layers[-1].get_weights()[0]
gap_weights.shape

cam_model = models.Model(inputs=model.inputs, outputs=(model.layers[-3].output, model.layers[-1].output))
feature_maps, predicted = cam_model.predict(X_test)
#================================================================
n_classes = Y_test.shape[1]
class_labels = np.array(dataset_dict[dataset_name[set_type]]['class_labels'])
true_label = np.argmax(Y_test, axis = 1)
pred_label = np.argmax(predicted, axis = 1)

for c in range(n_classes):
    plt.figure()
    index = np.where(true_label==c)[0]
    for idx in index:
        if pred_label[idx] == true_label[idx]:
            cam = np.zeros(dtype=np.float, shape=(feature_maps.shape[1]))
            for k, w in enumerate(gap_weights[:, true_label[idx] ]):
                cam += w * feature_maps[idx, :, k]
            cam = cam - np.min(cam)
            cam = cam / max(cam)
            cam = cam * 100
            t = np.linspace(1, feature_maps.shape[1], feature_maps.shape[1])
            y = X_test[idx, :, 0] * std + mean
            # linear interpolation to smooth
            max_length = 1000
            t_interp = np.linspace(1, feature_maps.shape[1], max_length, endpoint=True)
            y_interp = np.interp(t_interp, t, y)
            cam_interp = np.interp(t_interp, t, cam)
            cam_interp = cam_interp.astype(int)
            plt.scatter(x=t_interp, y=y_interp, c=cam_interp, cmap='jet', marker='.', s=1, vmin=0, vmax = 100)
    cbar = plt.colorbar()
    plt.savefig(path + 'results/Class_activation_map/CAM_'+ 'class_'+ str(int(c)) + '_' + filenames[set_type] + classifier[method] + '.pdf', bbox_inches='tight', dpi=1080)
#=================================================================
