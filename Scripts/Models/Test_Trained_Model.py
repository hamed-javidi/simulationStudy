# train and test MLP
#train and test TSF+MLP

import pandas as pd
import numpy as np
import os

path = '/home/khademg/simulationStudy/scripts/'
os.chdir(path)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
tf.keras.backend.clear_session()

import TSC_utils
from TSC_utils import directory_contents
from TSC_utils import read_dataset
from TSC_utils import train_test_dataset
from TSC_utils import calculate_metrics
from TSC_utils import plot_epochs_metric

# import MLP
# import TSF_MLP
from MLP import MLP_classifier
from TSF_MLP import TSF_MLP_classifier
from TSF_MLP import TSF_dataset
#======================================================================================================
# Select GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Fix random seed for reproducibility
seed = 1368
np.random.seed(seed)
#======================================================================================================
sel = 1                                 # choose classifier
nb_classes = 5
flag = False

root_path = '/home/khademg/simulationStudy/data/mag_cohorts/'
save_path = '/home/khademg/simulationStudy/scripts/results/'
directory_names = directory_contents(root_path)
classifiers = ['MLP', 'TSF_MLP']

directory_subset = directory_names[0:3]
    
for dirs in directory_subset:
    run_path = root_path + dirs
    fnames = directory_contents(run_path, 1)
    for fname in fnames:
        X, y = read_dataset(run_path, fname)
        x_train, y_train, x_val, y_val = train_test_dataset(X, y)
        save_name = dirs + fname[-11:-4]
        print(save_name)
        #####
        file_path = save_path + save_name + '_MLP_best_model.hdf5'
        model = models.load_model(file_path)
        y_pred = model.predict(x_val)
        #####
        file_path = save_path + save_name + '_TSF_MLP_best_model.hdf5'
        model = models.load_model(file_path)
        x_train, x_val = TSF_dataset(x_train, x_val)
        y_pred_TSF = model.predict(x_val)
        res = calculate_metrics(y_val, y_pred, 1, save_name)
        res_TSF = calculate_metrics(y_val, y_pred_TSF, 1, save_name)
        if flag == True:
            df_metrics = pd.concat((df_metrics, res), axis=0, sort=False)
            df_metrics_TSF = pd.concat((df_metrics_TSF, res_TSF), axis=0, sort=False)
        else:
            df_metrics = res
            df_metrics_TSF = res_TSF
            flag = True
