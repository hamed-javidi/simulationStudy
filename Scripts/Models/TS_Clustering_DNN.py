import sys

import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn import mixture

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from TSC_utils import *
from CNN_MLP_Clustering import *
from RNN_Clustering import *
from classifiers_dataset_info import *

# ======================================================================================================
def run_main_code(clustering, encoder):
    # List of autoencoders
    encoders_name = ['CNN_MLP', 'RNN']
    # List of clustering methods
    clustering_name = ['kmeans', 'GMM']
    # List of data sets
    dataset_names = ['mag_cohorts_2021', 'shape_cohorts_2021']
    # Number of Monte-Carlo runs
    n_runs = 1
    flag = False  # Always False
    seed = 1368
    # ======================================================================================================
    # Main Program - Model Training and Testing
    init_acc_eff = np.empty([0, 4])  # record loss over random initialization
    for dataset_name in dataset_names:
        nb_classes = dataset_dict[dataset_name]['nb_classes']
        root_path = dataset_dict[dataset_name]['path']
        dataset_length = sum([len(files) for r, d, files in os.walk(root_path)])
        processed_files=0
        directory_names = directory_contents(root_path)
        for dirs in directory_names:
            run_path = root_path + dirs
            fnames = directory_contents(run_path, 1)
            for fname in fnames:
                X, y, ID_Class = read_dataset(run_path, fname, dataset_dict, dataset_name)
                x_train, y_train, x_test, y_test, indexes = train_test_dataset(X, y, seed, slice_ratio = 0.5)
                X_in = np.concatenate((x_train, x_test))
                Y_in = np.concatenate((y_train, y_test))
                Y_in = np.argmax(Y_in, axis=1)
                save_name = dataset_name + '_' + dirs + fname[17:-4]
                print(save_name)
                MC_out = np.empty([0, 3])
                for run in range(n_runs):
                    print('MC Run #' + str(run))
                    file_path = save_path + 'models/' + save_name + '_' + encoders_name[encoder] + '_' + \
                                clustering_name[clustering] + '_Clustering_run_' + str(run) + '.hdf5'

                    if encoder == 0:  # select CNN_MLP
                        loss, duration, training_itrs = CNN_MLP_Clustering(save_path, save_name, X_in, nb_classes, run, file_path, nb_epochs=500)
                    elif encoder == 1:  # select RNN
                        loss, duration, training_itrs = RNN_Clustering(save_path, save_name, X_in, nb_classes, run, file_path, nb_epochs=500)
                    else:
                        sys.exit('Error!')
                    MC_out = np.vstack((MC_out, np.array([loss, duration, training_itrs])))
                median_index = np.argsort(MC_out[:, 0])[0]  # lowest loss
                #median_index = np.argsort(MC_out[:, 0])[len(MC_out[:, 0]) // 2]  # take median
                init_acc_eff = np.vstack((init_acc_eff, np.array([np.mean(MC_out[:, 0]), np.median(MC_out[:, 0]),
                                                                  np.min(MC_out[:, 0]), np.max(MC_out[:, 0])])))
                for run in range(n_runs):
                    if run == median_index:
                        src = save_path + 'models/' + save_name + '_' + encoders_name[encoder] + '_' + clustering_name[
                            clustering] + '_Clustering_run_' + str(run) + '.hdf5'
                        dst = save_path + 'models/' + save_name + '_' + encoders_name[encoder] + '_' + clustering_name[
                            clustering] + '_Clustering.hdf5'
                        os.rename(src, dst)
                        # Load results of median model
                        autoencoder = models.load_model(dst)
                        if encoder == 0:  # CNN_MLP
                            encoder_model = models.Model(inputs=autoencoder.inputs,
                                                         outputs=autoencoder.layers[12].output)
                        else:  # RNN
                            encoder_model = models.Model(inputs=autoencoder.inputs,
                                                         outputs=autoencoder.layers[3].output)
                        # Encode input data
                        X_in = encoder_model.predict(X_in)
                        # Select clustering method
                        if clustering == 0:  # Select k_means
                            start_time = time.time()
                            kmeans = KMeans(n_clusters=nb_classes).fit(X_in)
                            duration = time.time() - start_time
                            labels = kmeans.labels_
                            itr = kmeans.n_iter_
                            clusters = kmeans.cluster_centers_
                            cost = kmeans.inertia_  # Sum of squared distances of samples to their closest cluster center
                            cluster_membership = ['-' for x in range(len(X_in))]
                        else:  # Select GMM
                            start_time = time.time()
                            gmm = mixture.GaussianMixture(n_components=nb_classes, covariance_type='full').fit(X_in)
                            duration = time.time() - start_time
                            labels = gmm.predict(X_in)
                            itr = gmm.n_iter_
                            clusters = gmm.means_
                            cost = '-'
                            # Predict posterior probability of each component (cluster) given the data
                            cluster_membership = gmm.predict_proba(X_in)
                        if X_in.shape[-1] == 2:  # Plot results
                            clustering_viz(X_in, Y_in, clusters, labels, save_path, save_name, encoders_name[encoder],
                                           clustering_name[clustering])

                        cohort_path_name = '/' + dataset_name + '/' + dirs + '/' + fname
                        OUTPUTS = clustering_outputs(labels, cost, itr, cluster_membership, indexes, ID_Class, duration,
                                                     seed, cohort_path_name,
                                                     encoders_name[encoder], clustering_name[clustering], dataset_name,
                                                     dataset_dict)
                        if flag == True:
                            df_outputs = pd.concat((df_outputs, OUTPUTS), axis=0, sort=False)
                        else:
                            df_outputs = OUTPUTS
                            flag = True
                    else:
                        src = save_path + 'models/' + save_name + '_' + encoders_name[encoder] + '_' + clustering_name[
                            clustering] + '_Clustering_run_' + str(run) + '.hdf5'
                        os.remove(src)
                processed_files+=1
                print('-' * 40)
                print(f"Dataset: {dataset_name}")
                print(f"Progress: {processed_files} out of {dataset_length}")
                print('-' * 40)

    pd.DataFrame(init_acc_eff, columns=['MEAN', 'MED', 'MIN', 'MAX']).to_csv(
        save_path + encoders_name[encoder] + '_' + clustering_name[clustering] + '_initialization_effect.csv', index=False)

    df_outputs.to_csv(save_path + encoders_name[encoder] + '_' + clustering_name[clustering] + '_outputs.csv', index=False)


# ======================================================================================================
if __name__ == "__main__":
    path = os.path.abspath(os.getcwd()) + '/'
    save_path = path + 'results/'
    os.chdir(path)
    encoders_name = ['CNN_MLP', 'RNN']
    clustering_name = ['kmeans', 'GMM']
    for i in range(len(encoders_name)):
        print(f"{i}: {encoders_name[i]}")
    selected_encoder = int(input("Enter a number:"))
    for i in range(len(clustering_name)):
        print(f"{i}: {clustering_name[i]}")
    selected_clutering = int(input("Enter a number:"))
    core = input("Which GPU (0 to 7)?")
    # Select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(core)
    # train_percentile = [0.5, 0.3, 0.15]

    start = time.time()
    run_main_code(selected_clutering, selected_encoder)
    print(f"Elapsed time:  {time.time() - start}")

# ======================================================================================================
