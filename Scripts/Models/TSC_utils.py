# Timeseries Classification / Clustering

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc, average_precision_score

from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_curve
from numpy import random
from random import sample
import matplotlib.pyplot as plt
from manual_early_stop import TerminateOnBaseline
import os
import math
import pandas as pd
import numpy as np
from os import walk
import keras_tuner as kt



# ======================================================================================================

def directory_contents(path, flag=0):
    f_dirnames = []
    f_filenames = []
    for (dirpath, dirnames, filenames) in walk(path):
        f_dirnames.extend(dirnames)
        f_filenames.extend(filenames)
        break

    f_filenames = [f for f in f_filenames if (not f[0] == '.') & (f[-3:] == 'csv')]
    f_dirnames[:] = [d for d in f_dirnames if not d[0] == '.']

    if flag == 0:
        return sorted(f_dirnames)
    if flag == 1:
        return sorted(f_filenames)
    # ======================================================================================================


def tune_model(MyHyperModel, save_path, x_train, y_train, seed, project_name, params):
    tuner = kt.Hyperband(MyHyperModel(),
                     objective='val_loss',
                     max_epochs=30,
                     hyperband_iterations=1,
                     # num_initial_points=2,
                     seed=seed,
                     tune_new_entries=True,
                     allow_new_entries=True,
                     factor=3,
                     overwrite=False,
                     directory=save_path + 'tuned_models/',
                     project_name= project_name)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, min_delta=params['min_exp_val_loss'])
    baseline_stop = TerminateOnBaseline(monitor='val_acc', baseline=1.0, patience=5)
    tuner.search(x_train, y_train, epochs=30, validation_split=0.2, callbacks=[stop_early, baseline_stop], shuffle=True, use_multiprocessing=True, workers=48)
    # Get the optimal hyperparameters
    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hps = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters
    print(f"""
    Optimal values: {best_hps.values}
    """)
    return tuner, best_hps


def FillNAs(data, col_name):
    d = pd.DataFrame(data)
    d.fillna(method='backfill', inplace=True)
    d.fillna(method='ffill', inplace=True)
    return d[col_name]


def read_dataset(path_dir, filename, dataset_dict, dataset_name, vital_name, imb_ratio, scale='normalize'):
    path = path_dir + '/' + filename
    data = pd.read_csv(path)
    data['Class'] = data['Class'].str.lower()
    n_class = dataset_dict[dataset_name]['nb_classes']
    n_channels = dataset_dict[dataset_name]['n_channels']
    class_labels = np.char.lower(np.array(dataset_dict[dataset_name]['class_labels']))
    # To correct dataset column naming issue

    # Apply class imbalancy
    data = MakeClassesImbalanced(data, imb_ratio, class_labels, n_class)

    sample_size = 0
    for i in range(n_class):
        sample_size += data.loc[data['Class'] == class_labels[i], 'ID'].nunique()

    ts_l = int(round(len(data.ID) / sample_size))  # Length of time series

    X = np.empty((sample_size, ts_l, n_channels))
    y = np.empty((sample_size, n_class), dtype=int)
    ID = ["" for x in range(sample_size)]
    original_class = ["" for x in range(sample_size)]
    # for i in range(sample_size):
    #     slice= data.BMI[i * ts_l: (i + 1) * ts_l]
    #     if slice.isna().sum() > 0:
    #         data.BMI[i * ts_l: (i + 1) * ts_l] = FillNAs(slice)
    #     X[i, :, 0] =slice

    if scale == 'normalize':
        mean = np.mean(data[vital_name], axis=0)
        std = np.std(data[vital_name], axis=0)
        data[vital_name] -= mean
        data[vital_name] /= std
    elif scale == 'min_max':
        MIN = np.min(data[vital_name], axis=0)
        MAX = np.max(data[vital_name], axis=0)
        data[vital_name] = (2 * data[vital_name] - MAX - MIN) / (MAX - MIN)
        # Floating point inaccuracy!
        data[vital_name] = np.where(data[vital_name] >= 1., 1., data[vital_name])
        data[vital_name] = np.where(data[vital_name] <= -1., -1., data[vital_name])
    for i in range(sample_size):
        slice = data.loc[i * ts_l: (i + 1) * ts_l - 1, vital_name]
        if slice.isna().sum() > 0:
            slice = FillNAs(slice, vital_name)
        X[i, 0:ts_l, 0] = slice
        h = np.where(class_labels == data.loc[i * ts_l, 'Class'])
        y[i,] = to_categorical(h[0], n_class)
        ID[i] = data.loc[i * ts_l, 'ID']
        original_class[i] = data.loc[i * ts_l, 'Class']
    ID_Class = np.column_stack((ID, original_class))
    return X, y, ID_Class


# ======================================================================================================

def train_test_dataset(dataset, labels, seed, slice_ratio=0.5, shuffle=True):
    indexes = np.arange(len(dataset))
    if shuffle == True:
        # Fix random seed for reproducibility
        np.random.seed(seed)
        np.random.shuffle(indexes)
    n_sample_train = int(np.round(len(dataset) * slice_ratio[0]))
    n_sample_val = int(np.round(len(dataset) * (slice_ratio[0] + slice_ratio[1])))

    train_X = dataset[indexes[:n_sample_train]]
    train_Y = labels[indexes[:n_sample_train]]

    val_X = dataset[indexes[n_sample_train:n_sample_val]]
    val_Y = labels[indexes[n_sample_train:n_sample_val]]

    test_X = dataset[indexes[n_sample_val:]]
    test_Y = labels[indexes[n_sample_val:]]
    return train_X, train_Y, val_X, val_Y, test_X, test_Y, indexes


# ======================================================================================================
def MakeClassesImbalanced(data, ratio, class_labels, n_classes):
    if n_classes != 2:
        import sys
        sys.exit('Number of classes is > 2')
    cases = data.loc[data['Class'] == class_labels[1], :]
    controls = data.loc[data['Class'] == class_labels[0], :]
    cases_ids = cases['ID'].nunique()
    new_cases_pop = int(controls['ID'].nunique() * ratio)
    rnd_selected_cases = sample(cases['ID'].unique().tolist(), new_cases_pop)
    new_cases = cases.loc[cases['ID'].isin(rnd_selected_cases), :]
    data = controls.append(new_cases)
    data.reset_index(inplace=True)
    if data.loc[data['Class'] == class_labels[1], 'ID'].nunique() != ratio * data.loc[data['Class'] == class_labels[0], 'ID'].nunique():
        import sys
        sys.exit('Something wrong in MakeClassesImbalanced!')
    return data


# ======================================================================================================

def calculate_metrics(y_test, y_test_pred, filename, train_auc, val_auc, test_auc, best_thresh, accuracy, duration, training_itrs):
    y_test = np.argmax(y_test, axis=1)
    y_test_pred = y_test_pred[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred, pos_label=1)
    ix = np.argmax(tpr - fpr)

    # AUPRC calculation
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_pred)

    y_test_pred = (y_test_pred > best_thresh).astype("int")
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    except ValueError:
        tn = fp = fn = tp = 1
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    res = pd.DataFrame(index=[0])
    res['cohort_name'] = filename
    res['Specificity'] = round(tn / (tn + fp), 2)
    res['Sensitivity'] = round(tp / (tp + fn), 2)
    res['balanced_test_accuracy'] = round(balanced_accuracy_score(y_test, y_test_pred), 2)
    res['train_AUC'] = train_auc
    res['val_AUC'] = val_auc
    res['test_AUC'] = test_auc
    res['test_accuracy'] = accuracy
    res['best_thresh'] = best_thresh
    res['AUPRC'] = auc(recall, precision)
    res['duration'] = round(duration / 60, 3)  # in minutes
    res['iteration'] = training_itrs
    res['TP'] = tp
    res['TN'] = tn
    res['FP'] = fp
    res['FN'] = fn
    res['0: precision'] = report_dict['0']['precision']
    res['0: recall'] = report_dict['0']['recall']
    res['0: f1_score'] = report_dict['0']['f1-score']
    res['0: support'] = report_dict['0']['support']

    res['1: precision'] = report_dict['1']['precision']
    res['1: recall'] = report_dict['1']['recall']
    res['1: f1_score'] = report_dict['1']['f1-score']
    res['1: support'] = report_dict['1']['support']

    res['macro-avg: precision'] = report_dict['macro avg']['precision']
    res['macro-avg: recall'] = report_dict['macro avg']['recall']
    res['macro-avg: f1_score'] = report_dict['macro avg']['f1-score']
    res['macro-avg: support'] = report_dict['macro avg']['support']

    res['weighted-avg: precision'] = report_dict['weighted avg']['precision']
    res['weighted-avg: recall'] = report_dict['weighted avg']['recall']
    res['weighted-avg: f1_score'] = report_dict['weighted avg']['f1-score']
    res['weighted-avg: support'] = report_dict['weighted avg']['support']
    return res


# ======================================================================================================

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    plt.close()


# ======================================================================================================
def prepare_outputs(y_train, y_train_pred, y_val, y_val_pred, duration, training_itrs,
                    indexes, ID_Class, seed, filename, classifier_name, dataset_name, dataset_dict):  # method_dict,
    training_time = [round(duration / 60, 3) for x in range(len(y_train) + len(y_val))]
    training_iteration = [training_itrs for x in range(len(y_train) + len(y_val))]
    random_seed = [seed for x in range(len(y_train) + len(y_val))]
    num_k = ['-' for x in range(len(y_train) + len(y_val))]
    cohort_name = [filename for x in range(len(y_train) + len(y_val))]
    method = [classifier_name for x in range(len(y_train) + len(y_val))]
    # parameter_names = [method_dict[classifier_name]['params'] for x in range(len(y_train)+len(y_val))]
    # parameter_values = [method_dict[classifier_name]['values'] for x in range(len(y_train)+len(y_val))]
    prob_each_class_order = [dataset_dict[dataset_name]['class_order'] for x in range(len(y_train) + len(y_val))]
    train_test = ['Train' for x in range(len(y_train))] + ['Test' for x in range(len(y_val))]
    y_train_pred_label = np.argmax(y_train_pred, axis=1)
    y_val_pred_label = np.argmax(y_val_pred, axis=1)
    class_labels = dataset_dict[dataset_name]['class_labels']
    predicted_class = [class_labels[x] for x in y_train_pred_label] + [class_labels[x] for x in y_val_pred_label]
    y_train_test_pred = np.row_stack((y_train_pred, y_val_pred))
    prob_each_class = ['' for x in range(len(y_train_test_pred))]
    j = 0
    for x in y_train_test_pred:
        A = ''
        for i in range(len(class_labels)):
            if i < len(class_labels) - 1:
                A += str(round(x[i], 3)) + ' | '
            else:
                A += str(round(x[i], 3))
        prob_each_class[j] = A
        j += 1
    ID_Class_sorted = ID_Class[indexes]
    data_frame = {'id': ID_Class_sorted[:, 0], 'original_class': ID_Class_sorted[:, 1], 'training_time': training_time,
                  'training_iteration': training_iteration, 'predicted_class': predicted_class,
                  'cohort_name': cohort_name,
                  'method': method, 'random_seed': random_seed, 'train_test': train_test, 'num_k': num_k,
                  'prob_each_class': prob_each_class,
                  'prob_each_class_order': prob_each_class_order}  # 'parameter_names':parameter_names, 'parameter_values':parameter_values,
    df = pd.DataFrame(data_frame, index=indexes)
    df = df.sort_index(axis=0)
    return df


# ======================================================================================================

def gramian_angular_field(dataset):
    n_samples, series_len, n_channels = dataset.shape
    max_img_size = 128
    if (series_len <= max_img_size):
        img_size = series_len
        PAA_flag = False
    else:
        img_size = max_img_size
        PAA_flag = True
    X_GAF = np.empty((n_samples, img_size, img_size, n_channels))
    for i in range(n_samples):
        for j in range(n_channels):
            if (PAA_flag == True):
                serie = PAA(dataset[i, :, j], max_img_size)
            else:
                serie = dataset[i, :, j]
            # Polar encoding
            phi = np.arccos(serie)
            # Note! The computation of r is not necessary
            # r = np.linspace(0, 1, len(serie))
            # GAF Computation
            X_GAF[i, :, :, j] = np.vectorize(cos_sum)(*np.meshgrid(phi, phi, sparse=True))
    return X_GAF


def cos_sum(a, b):
    return (math.cos(a + b))


def PAA(serie, output_serie_len):
    # Piecewise Aggregate Approximation
    print('Performing Piecewise Aggregate Approximation ...')
    if (len(serie) % output_serie_len == 0):
        splitted = np.array_split(serie, output_serie_len)
        out = [item.mean() for item in splitted]
    else:
        value_space = np.arange(0, len(serie) * output_serie_len)
        output_index = value_space // len(serie)
        input_index = value_space // output_serie_len
        uniques, nUniques = np.unique(output_index, return_counts=True)
        out = [serie[indices].sum() / len(serie) for indices in
               np.split(input_index, nUniques.cumsum())[:-1]]
    # ======================================================================================================


def GAF_viz(X_GAF, Y_GAF, save_path, save_name, dataset_name, dataset_dict):
    n_samples, width, height, channels = X_GAF.shape
    Y_GAF = np.argmax(Y_GAF, axis=1)
    uniques = np.unique(Y_GAF)
    for k in range(len(uniques)):
        X = X_GAF[np.where(Y_GAF == uniques[k])]
        class_mode = dataset_dict[dataset_name]['class_labels'][uniques[k]]
        for i in range(channels):
            data_plt = np.mean(X[:, :, :, i], axis=0)
            img = plt.imshow(data_plt, interpolation='nearest')
            plt.axis('off')
            plt.savefig(save_path + 'GAF/' + save_name + '_' + class_mode + '_channel_' + str(i) + '.pdf',
                        bbox_inches='tight')


# =======================================================================================================

def clustering_outputs(labels, cost, itr, cluster_membership, indexes, ID_Class, duration, seed, filename, encoder_name,
                       clustering_name, dataset_name, dataset_dict):
    random_seed = [seed for x in range(len(labels))]
    cohort_name = [filename for x in range(len(labels))]
    encoder_method = [encoder_name for x in range(len(labels))]
    clustering_method = [clustering_name for x in range(len(labels))]
    distance_cluster_centers = [cost for x in range(len(labels))]
    clustering_time = [duration for x in range(len(labels))]
    clustering_itr = [itr for x in range(len(labels))]
    class_labels = dataset_dict[dataset_name]['class_labels']
    cluster_memberships = [str(x) for x in cluster_membership]
    ID_Class_sorted = ID_Class[indexes]
    data_frame = {'id': ID_Class_sorted[:, 0], 'original_class': ID_Class_sorted[:, 1], 'predicted_cluster': labels,
                  'distance_to_cluster_centers': distance_cluster_centers, 'clustering_time': clustering_time,
                  'clustering_iteration': clustering_itr, 'cohort_name': cohort_name, 'encoder_method': encoder_method,
                  'clustering_method': clustering_method, 'random_seed': random_seed}
    df = pd.DataFrame(data_frame, index=indexes)
    df = df.sort_index(axis=0)
    return df


# =======================================================================================================

def clustering_viz(X, Y, mu, cluster_labels, save_path, save_name, encoder_name, clustering_name):
    Y = Y.astype(int)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.xlabel('latent_dim1')
    plt.ylabel('latent_dim2')
    plt.title('Ground Truth', size=15)
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels)
    plt.scatter(mu[:, 0], mu[:, 1], c='black', marker='x')
    plt.xlabel('latent_dim1')
    # plt.ylabel('latent_dim2')
    plt.title('Clustering', size=15)
    plt.savefig(save_path + 'clustering/' + save_name + '_' + encoder_name + '_' + clustering_name + '_latent_space.pdf', bbox_inches='tight')


# =======================================================================================================

def GetBestThreshold(y_test, y_test_pred):
    fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), y_test_pred[:, 1], pos_label=1)
    x = tpr - fpr
    sorted_x = np.argsort(-x, kind='quicksort')
    found=0
    if 0 < thresholds[sorted_x[0]] < 1:
        best_thrsh = thresholds[sorted_x[0]]
    else:
        for i in range(1, len(sorted_x)):
            if 0 < thresholds[sorted_x[i]] < 1:
                best_thrsh = thresholds[sorted_x[i]]
                found = 1
        if found==0:
            import sys
            print(f'Threshold issue!; thresh[{i}]= {thresholds[sorted_x[i]]}')
            best_thrsh=0.5
    return best_thrsh
