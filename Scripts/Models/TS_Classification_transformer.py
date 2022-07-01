import sys

import pandas as pd
import numpy as np
import os
from pathlib import Path
import time

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from TSC_utils import *
from MLP import *
from TSF_MLP import *
from TSF_CNN import *
from GAF_CNN import *
from FCNN import *
from ResNet import *
from RNN import *
from LSTMFCN import *
from CNN_RNN import *
from CNN_MLP_Autoencoder import *
from RNN_Autoencoder import *
from ts_transformer import *
from classifiers_dataset_info import *
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# ======================================================================================================
def run_main_code(selected_classifier, seed, test_frac, verbose=False, k_fold=3):
    scale = {
        'MLP': 'normalize',
        'TSF_MLP': 'normalize',
        'GAF_CNN': 'min_max',
        'FCNN': 'normalize',
        'ResNet': 'normalize',
        'RNN': 'normalize',
        'CNN_RNN': 'normalize',
        'CNN_MLP_Autoencoder': 'normalize',
        'RNN_Autoencoder': 'normalize',
        'TSF_CNN': 'normalize',
        'LSTMFCN': 'normalize',
        'Transformer': 'normalize'
    }

    processed_files = 0
    df_metrics = []
    # df_outputs = []
    # init_acc_eff = np.empty([0, 4])
    # auc_report = np.empty([0, 4])
    # cohort_names = directory_contents(root_path)
    # for cohort_name in cohort_names:
    #     patient_cohorts = directory_contents(root_path + cohort_name + '/')
    #     patient_cohorts.remove('hyperparameter_cohorts')

        # for patient_cohort in patient_cohorts:

    # To recover existing results
    if recovery_mode:
        path = save_path + selected_classifier + '_' + vital_type + '_' + cohort_name + '_' + patient_cohort + '_metrics.csv'
        recovered_results = pd.read_csv(path)
    sub_cohorts = directory_contents(root_path)

    for sub_cohort in sub_cohorts:
        fnames = directory_contents(root_path + sub_cohort + '/', 1)
        file_path = root_path + sub_cohort

        for fname in fnames:
            print()
            print(f'Path: {file_path}'
                  f'File name: {fname}')

            # Loop over imbalancing ratio
            for imb_ratio in [0.25, 0.5, 0.75, 1]:
                arr = fname.split('_')
                mean = arr[4]
                std = arr[5]
                missingness = arr[6]
                irregularity = arr[7]
                overlaping = arr[8][:-4]
                save_name = dataset_name + '_' + vital_type + '_' + cohort_name + '_' + patient_cohort + '_' + mean + '_' + std + '_' + missingness + '_' + irregularity + '_' + str(
                    imb_ratio) + '_' + overlaping
                print(f'Save name: {save_name}')

                if recovery_mode:
                    if save_name in recovered_results['cohort_name'].values:
                        df_metrics.append(recovered_results.loc[recovered_results['cohort_name'] == save_name, :])
                        continue
                X, y, ID_Class = read_dataset(file_path, fname, dataset_dict, dataset_name, vital_type.split('_')[0], imb_ratio,
                                              scale=scale[selected_classifier])
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, stratify=y, random_state=seed)

                # Balanced train set
                if imb_ratio != 1:
                    p_idx = np.where((y_train[:, 1]) == 1)
                    n_idx = np.where((y_train[:, 1]) == 0)
                    X_p = x_train[p_idx]
                    X_n = x_train[n_idx]
                    y_p = y_train[p_idx]
                    y_n = y_train[n_idx]
                    y_train = np.vstack((y_p, y_n[1:len(y_p) + 1, :]))
                    x_train = np.vstack((X_p, X_n[1:len(X_p) + 1, :, :]))
                    x_train, y_train = shuffle(x_train, y_train)

                assert sum((y_train[:, 1]) == 1) == sum((y_train[:, 1]) == 0)

                params = {
                    "save_path": save_path,
                    "save_name": save_name,
                    "vital_type": vital_type,
                    "cohort_type": cohort_name,
                    "mean": mean,
                    "std": std,
                    "irregularity": irregularity,
                    "missingness": missingness,
                    "imb_ratio": imb_ratio,
                    "nb_classes": nb_classes,
                    "seed": seed,
                    "verbose": verbose,
                    "min_exp_val_loss": min_exp_val_loss,
                    "folds": k_fold,
                    "tune_project_name": vital_type + '_' + cohort_name + '_' + mean + '_' + std + '_' + patient_cohort + '_' + irregularity
                }

                # Hold model outputs on train and test data sets during Monte Carlo runs
                # out_res_train = np.empty([n_runs, y_train.shape[0], y_train.shape[1]])
                # out_res_test = np.empty([n_runs, y_test.shape[0], y_test.shape[1]])

                if selected_classifier == 'MLP':
                    y_train_pred, y_val_pred, y_test_pred, metric = MLP_classifier(x_train, y_train, x_test, y_test, params)

                elif selected_classifier == 'TSF_MLP':
                    y_train_pred, y_val_pred, y_test_pred, metric = TSF_MLP_classifier(x_train, y_train, x_test, y_test, params)

                elif selected_classifier == 'GAF_CNN':
                    x_train = gramian_angular_field(x_train)
                    x_test = gramian_angular_field(x_test)
                    # X_GAF = np.concatenate((x_train, x_test), axis=0)
                    # Y_GAF = np.concatenate((y_train, y_test), axis=0)
                    # GAF_viz(X_GAF, Y_GAF, save_path, save_name, dataset_name, dataset_dict)  # GAF Visualization
                    y_train_pred, y_val_pred, y_test_pred, metric = GAF_CNN_classifier(x_train, y_train, x_test, y_test, params)

                elif selected_classifier == 'FCNN':
                    y_train_pred, y_val_pred, y_test_pred, metric = FCNN_classifier(x_train, y_train, x_test, y_test, params)

                elif selected_classifier == 'ResNet':
                    y_train_pred, y_val_pred, y_test_pred, metric = ResNet_classifier(x_train, y_train, x_test, y_test, params)

                elif selected_classifier == 'RNN':
                    y_train_pred, y_val_pred, y_test_pred, metric = RNN_classifier(x_train, y_train, x_test, y_test, params)

                elif selected_classifier == 'CNN_RNN':
                    y_train_pred, y_val_pred, y_test_pred, metric = CNN_RNN_classifier(x_train, y_train, x_test, y_test, params)

                elif selected_classifier == 'CNN_MLP_Autoencoder':
                    y_train_pred, y_val_pred, y_test_pred, metric = CNN_MLP_Autoencoder(x_train, y_train, x_test, y_test, params)

                elif selected_classifier == 'RNN_Autoencoder':
                    y_train_pred, y_val_pred, y_test_pred, metric = RNN_Autoencoder(x_train, y_train, x_test, y_test, params)

                elif selected_classifier == 'TSF_CNN':
                    y_train_pred, y_val_pred, y_test_pred, metric = TSF_CNN_classifier(x_train, y_train, x_test, y_test, params)
                elif selected_classifier == 'LSTMFCN':
                    y_train_pred, y_val_pred, y_test_pred, metric = LSTMFCN_classifier(x_train, y_train, x_test, y_test, params)

                elif selected_classifier == 'Transformer':
                    y_train_pred, y_val_pred, y_test_pred, metric = Transformer_classifier(x_train, y_train, x_test, y_test, params)

                else:
                    sys.exit('Selected classifier does not exist!')
                # # Decision threshold is calculated based on Youden’s J statistic.
                # # J = Sensitivity + (1 – FalsePositiveRate) – 1
                # # Which we can restate as:
                # # J = TruePositiveRate – FalsePositiveRate
                # fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), y_test_pred[:, 1], pos_label=1)
                # test_AUC = roc_auc_score(np.argmax(y_test, axis=1), y_test_pred[:, 1])
                # # if test_AUC < 0.4:
                # #     raise Exception(f"Test AUC is wierd! : {test_AUC}")
                #
                # # get the best threshold
                # ix = np.argmax(tpr - fpr)
                # best_thresh = thresholds[ix] if thresholds[ix] < 1 else thresholds[ix + 1]
                # if 1 < best_thresh < 0:
                #     raise Exception(f"Decision threshold issue! : {best_thresh}")
                # acc_best = accuracy_score(np.argmax(y_test, axis=1), (y_test_pred[:, 1] >= best_thresh).astype("int"))
                #
                # fpr, tpr, thresholds = roc_curve(np.argmax(y_train, axis=1), y_train_pred[:, 1], pos_label=1)
                # train_AUC = roc_auc_score(np.argmax(y_train, axis=1), y_train_pred[:, 1])
                # print(f'Train AUC:{round(train_AUC, 2)}, Test AUC: {round(test_AUC, 2)}, Optimal threshold: {best_thresh}')
                # MC_out = np.array([train_AUC, test_AUC, best_thresh, acc_best, duration, training_itrs])
                # MC_out = pd.DataFrame(MC_out, columns=['train_AUC', 'test_AUC', 'best_thresh', 'acc_best', 'duration', 'training_itrs'])
                # cohort_path_name = '/' + dataset_name + '/' + dirs + '/' + fname
                # METRICS = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred, save_name, MC_out)
                df_metrics.append(metric)
                print(f""
                      f"Progress:{processed_files / total_files * 100}"
                      f"")
            processed_files = processed_files + 1
            if len(df_metrics) > 0:
                pd.concat(df_metrics, axis=0, sort=False).to_csv(save_path + selected_classifier + '_' + vital_type + '_' + cohort_name + '_' + patient_cohort + '_metrics.csv', index=False)
    if len(df_metrics) > 0:
        df_metrics = pd.concat(df_metrics, axis=0, sort=False)
        # df_outputs = pd.concat(df_outputs, axis=0, sort=False)
        # pd.DataFrame(init_acc_eff, columns=['MEAN', 'MED', 'MIN', 'MAX']).to_csv(save_path + selected_classifier + '_initialization_effect.csv', index=False)
        df_metrics.to_csv(save_path + selected_classifier + '_' + vital_type + '_' + cohort_name + '_' + patient_cohort + '_metrics.csv', index=False)
        # df_outputs.to_csv(save_path + selected_classifier + '_outputs.csv', index=False)


# ======================================================================================================
if __name__ == "__main__":
    path = os.path.abspath(os.getcwd()) + '/'
    save_path = path + 'results/'
    os.chdir(path)

    classifiers_name = ['MLP', 'TSF_MLP', 'GAF_CNN', 'FCNN', 'ResNet', 'CNN_RNN', 'TSF_CNN', 'LSTMFCN', 'Transformer']
    vital_types = ['bmi_simulation_clean', 'glucose_simulation_clean', 'sbp_simulation_clean']
    cohort_types = ['mag_cohorts', 'shape_cohorts']
    if len(sys.argv) > 1:
        print("Number of arguments:", len(sys.argv))
        print(f'Arg1: {sys.argv[1]}\n'
              f'Arg2: {sys.argv[2]}')
    # dataset_name = 'local_2022v1'
    dataset_name = '2022v1'

    classifier = classifiers_name[int(sys.argv[1])]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[3])
    num = int(sys.argv[2])
    cohort_type_num = int(num /  9)
    cohort_name = cohort_types[cohort_type_num]

    num = num % 9
    vital_type_num = int(num / 3)
    vital_type = vital_types[vital_type_num]

    selected_folder = num % 3


    nb_classes = dataset_dict[dataset_name]['nb_classes']
    root_path = dataset_dict[dataset_name]['path'] + vital_type + '/' + cohort_name + '/'
    patient_cohorts = directory_contents(root_path)
    patient_cohort = patient_cohorts[selected_folder]
    root_path = root_path + patient_cohort + '/'

    # classifier = 'ResNet'
    # cohort_type = 'mag_cohorts'
    seed = 1368
    test_size = 0.3
    n_cross_val_folds = 3
    tunning = 1
    min_exp_val_loss = 0.0001
    total_files = 176
    verbose = False
    recovery_mode = False  # To reuse results from the last unfinished run


    # print("Enter a number:")
    # for i in range(len(classifiers_name)):
    #     print(f"{i}: {classifiers_name[i]}")
    # num = int(input("Which Classifier (0 to 11)?"))
    # core = input("Which GPU (0 to 7)?")

    # Number of Monte-Carlo runs
    # n_runs = int(input("Number of run?"))

    # n_runs=10
    # Select GPU
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(core)
    print(f'Selected Classifier: {classifier}'
          f'Selected vital: {vital_type}'
          f'Selected cohort: {cohort_name}')
    start = time.time()
    run_main_code(classifier, seed, test_size, verbose=verbose, k_fold=n_cross_val_folds)
    print(f"Classifier: {classifier} Elapsed time:  {time.time() - start}")
    # for i in range(9,len(classifiers_name)):
    #     print(f"{i}: {classifiers_name[i]}")
    #     start = time.time()
    #     run_main_code(i, n_runs= n_runs, slice_ratio= slice_ratio)
    #     print(f"Classifier: {classifiers_name[i]} Elapsed time:  {time.time()-start}")
# ======================================================================================================
