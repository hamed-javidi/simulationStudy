import pandas as pd
import numpy as np
import os
from pathlib import Path
import time

from sklearn.metrics import accuracy_score
from TSC_utils import *
from MLP import *
from TSF_MLP import *
from GAF_CNN import *
from FCNN import *
from ResNet import *
from RNN import *
from CNN_RNN import *
from CNN_MLP_Autoencoder import *
from RNN_Autoencoder import *
from classifiers_dataset_info import *
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# List of data sets
dataset_names = ['mag_cohorts_2021_v2', 'shape_cohorts_2021_v2']

path = os.path.abspath(os.getcwd()) + '/'
save_path = path + 'results/'
os.chdir(path)



#======================================================================================================
def run_main_code(sel, n_runs = 1):
    # Data scaling method
    scale = ['normalize', 'normalize', 'min_max', 'normalize', 'normalize', 'normalize', 'normalize',
             'normalize', 'normalize']


    print(classifiers_name[sel] + ' is running ...')
    # Number of Monte-Carlo runs
    n_runs = 1
    flag = False     # Always False
    seed = 1368
    #======================================================================================================
    # Main Program - Model Training and Testing
    init_acc_eff = np.empty([0, 4])    # record accuracy over random initialization
    for dataset_name in dataset_names:
        print(f'Dataset: {dataset_name}')
        nb_classes = dataset_dict[dataset_name]['nb_classes']
        root_path = dataset_dict[dataset_name]['path']
        directory_names = directory_contents(root_path)
        for dirs in directory_names:
            run_path = root_path + dirs
            fnames = directory_contents(run_path, 1)
            for fname in fnames:
                X, y, ID_Class = read_dataset(run_path, fname, dataset_dict, dataset_name, scale = scale[sel])
                x_train, y_train, x_test, y_test, indexes = train_test_dataset(X, y, seed, slice_ratio = 0.7)
                save_name = dataset_name + '_' + dirs + fname[17:-4]
                print(run_path + " / " + fname)
                print(save_name)
                MC_out = np.empty([0, 5])
                # Hold model outputs on train and test data sets during Monte Carlo runs
                out_res_train = np.empty([n_runs, y_train.shape[0], y_train.shape[1]])
                out_res_test = np.empty([n_runs, y_test.shape[0], y_test.shape[1]])
                for run in range(n_runs):
                    print('MC Run #' + str(run))
                    if sel == 0:   # select MLP
                        y_train_pred, y_test_pred, duration, training_itrs = MLP_classifier(save_path, save_name,
                                                                                            x_train, y_train, x_test, y_test, nb_classes, run, min_exp_val_loss=0.02)
                    elif sel == 1: # select TSF_MLP
                        y_train_pred, y_test_pred, duration, training_itrs = TSF_MLP_classifier(save_path, save_name,
                                                                                                x_train, y_train, x_test, y_test, nb_classes, run, min_exp_val_loss=0.02)
                    elif sel == 2: # Select GAF_CNN
                        if run == 0:
                            x_train = gramian_angular_field(x_train)
                            x_test = gramian_angular_field(x_test)
                            # X_GAF = np.concatenate((x_train, x_test), axis=0)
                            # Y_GAF = np.concatenate((y_train, y_test), axis=0)
                            # GAF_viz(X_GAF, Y_GAF, save_path, save_name, dataset_name, dataset_dict)  # GAF Visualization
                        y_train_pred, y_test_pred, duration, training_itrs = GAF_CNN_classifier(save_path, save_name,
                                                                                                x_train, y_train, x_test, y_test, nb_classes, run, min_exp_val_loss=0.02)
                    elif sel == 3: # Select FCNN
                        y_train_pred, y_test_pred, duration, training_itrs = FCNN_classifier(save_path, save_name,
                                                                                             x_train, y_train, x_test, y_test, nb_classes, run, min_exp_val_loss=0.02)
                    elif sel == 4: # Select ResNet
                        y_train_pred, y_test_pred, duration, training_itrs = ResNet_classifier(save_path, save_name,
                                                                                               x_train, y_train, x_test, y_test, nb_classes, run, min_exp_val_loss=0.02)
                    elif sel == 5: # Select RNN
                        y_train_pred, y_test_pred, duration, training_itrs = RNN_classifier(save_path, save_name,
                                                                                               x_train, y_train, x_test, y_test, nb_classes, run, min_exp_val_loss=0.02)
                    elif sel == 6: # Select RNN
                        y_train_pred, y_test_pred, duration, training_itrs = CNN_RNN_classifier(save_path, save_name,
                                                                                               x_train, y_train, x_test, y_test, nb_classes, run, min_exp_val_loss=0.02)
                    elif sel == 7: # Select CNN_MLP_Autoencoder
                        y_train_pred, y_test_pred, duration, training_itrs = CNN_MLP_Autoencoder(save_path, save_name,
                                                                                               x_train, y_train, x_test, y_test, nb_classes, run, min_exp_val_loss=0.02)
                    elif sel == 8: # Select RNN_Autoencoder
                        y_train_pred, y_test_pred, duration, training_itrs = RNN_Autoencoder(save_path, save_name,
                                                                                               x_train, y_train, x_test, y_test, nb_classes, run, min_exp_val_loss=0.02)
                    else:
                        sys.exit('Error!')

                    fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_test_pred[:, 1])
                    # get the best threshold
                    J = tpr - fpr
                    ix = np.argmax(J)
                    best_thresh = thresholds[ix]
                    my_auc = auc(fpr, tpr)
                    acc_best = accuracy_score(np.argmax(y_test, axis=1), (y_test_pred[:, 1] >= best_thresh).astype("int"))
                    MC_out = np.vstack((MC_out, np.array([my_auc, best_thresh, acc_best, duration, training_itrs])))

                    out_res_train[run] = y_train_pred
                    out_res_test[run] = y_test_pred
                median_index = np.argsort(MC_out[:, 0])[len(MC_out[:, 0])//2]
                init_acc_eff = np.vstack((init_acc_eff, np.array([np.mean(MC_out[:, 0]), np.median(MC_out[:, 0]),
                                                                  np.min(MC_out[:, 0]), np.max(MC_out[:, 0])])))
                for run in range(n_runs):
                    if run == median_index:
                        src = save_path + 'models/' + save_name + '_' + classifiers_name[sel] + '_history_run_' + str(run) + '.csv'
                        dst = save_path + 'models/' + save_name + '_' + classifiers_name[sel] + '_history.csv'
                        os.rename(src, dst)
                        src = save_path + 'models/' + save_name + '_' + classifiers_name[sel] + '_best_model_run_' + str(run) + '.hdf5'
                        dst = save_path + 'models/' + save_name + '_' + classifiers_name[sel] + '_best_model.hdf5'
                        os.rename(src, dst)
                        # Load results of median model
                        y_test_pred = out_res_test[median_index]
                        y_train_pred = out_res_train[median_index]
                        cohort_path_name = '/' + dataset_name + '/' + dirs + '/' + fname
                        METRICS = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred, save_name, MC_out[median_index, :])
                        OUTPUTS = prepare_outputs(y_train, y_train_pred, y_test, y_test_pred, duration, training_itrs,
                                                  indexes, ID_Class, seed, cohort_path_name, classifiers_name[sel],
                                                  dataset_name, dataset_dict) # method_dict,
                        if flag == True:
                            df_metrics = pd.concat((df_metrics, METRICS), axis=0, sort=False)
                            df_outputs = pd.concat((df_outputs, OUTPUTS), axis=0, sort=False)
                        else:
                            df_metrics = METRICS
                            df_outputs = OUTPUTS
                            flag = True
                    else:
                        src = save_path + 'models/' + save_name + '_' + classifiers_name[sel] + '_best_model_run_' + str(run) + '.hdf5'
                        os.remove(src)
                        src = save_path + 'models/' + save_name + '_' + classifiers_name[sel] + '_history_run_' + str(run) + '.csv'
                        os.remove(src)


    pd.DataFrame(init_acc_eff, columns=['MEAN', 'MED', 'MIN', 'MAX']).to_csv(save_path + classifiers_name[sel] + '_initialization_effect.csv', index = False)
    df_metrics.to_csv(save_path + classifiers_name[sel] + '_metrics.csv', index = False)
    df_outputs.to_csv(save_path + classifiers_name[sel] + '_outputs.csv', index = False)
#======================================================================================================
if __name__ == "__main__":
    # List of classifiers

    classifiers_name = ['MLP', 'TSF_MLP', 'GAF_CNN', 'FCNN', 'ResNet', 'RNN', 'CNN_RNN', 'CNN_MLP_Autoencoder', 'RNN_Autoencoder']
    print("Enter a number:")
    for i in range(len(classifiers_name)):
        print(f"{i}: {classifiers_name[i]}")
    num = int(input())
    core = input("Which GPU (0 to 7)?")
    # Select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(core)

    start = time.time()
    run_main_code(num, n_runs= 1)
    print(f"Classifier: {classifiers_name[num]} Elapsed time:  {time.time()-start}")
#======================================================================================================
