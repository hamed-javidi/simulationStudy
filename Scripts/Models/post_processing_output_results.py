from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from numpy import random
import matplotlib.pyplot as plt

import os
import math
import pandas as pd
import numpy as np
from os import walk
#======================================================================================================

path = os.path.abspath(os.getcwd()) + '/results/'
save_path = path + 'post_process_results/'
os.chdir(path)

# Select classifier
sel = 0            

# List of classifiers
classifiers_name = ['MLP', 'TSF_MLP', 'GAF_CNN', 'FCNN', 'ResNet', 'RNN', 'CNN_RNN', 'CNN_MLP_Autoencoder', 'RNN_Autoencoder']

# List of data sets
dataset_names = ['mag_cohorts_2021', 'shape_cohorts_2021']
cohort_dict = {dataset_names[0]:['0', '0.1', '0.25', '0.5'], dataset_names[1]: ['0', '0.1', '0.25', '0.5']}
#======================================================================================================

output = pd.read_csv(path+classifiers_name+'_outputs.csv')

cohorts_names = np.unique(output.cohort_name)


for cohort in dataset_names:
    matching = [s for s in cohorts_names if cohort in s]
    first_indx = matching[0]
    end_indx = matching[-1]
    level = cohort_dict[cohort]
    for lvl in level:
        index_cohort = [s for s in cohorts_names[first_indx:end_indx] if lvl in s]



history = pd.read_csv(path+classifiers_name+'_outputs.csv')




y_train = np.argmax(y_train, axis=1)
    y_train_pred = np.argmax(y_train_pred, axis=1)
    y_val = np.argmax(y_val, axis=1)
    y_val_pred = np.argmax(y_val_pred, axis=1)
    res = pd.DataFrame(data = np.zeros((1,8),dtype=np.float), index=[0],
                       columns=['cohort_name','accuracy_train','precision_train', 'recall_train',
                                'accuracy_test','precision_test', 'recall_test','duration'])
    res['cohort_name'] = filename
    res['accuracy_train'] = accuracy_score(y_train, y_train_pred)
    res['precision_train'] = precision_score(y_train, y_train_pred, average='macro')
    res['recall_train'] = recall_score(y_train, y_train_pred, average='macro')
    res['accuracy_test'] = accuracy_score(y_val, y_val_pred)
    res['precision_test'] = precision_score(y_val, y_val_pred, average='macro')
    res['recall_test'] = recall_score(y_val, y_val_pred, average='macro')
    res['duration'] = round(duration/60, 3)   # in minutes
