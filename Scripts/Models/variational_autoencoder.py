# Data Generation Using Variational Autoencoder

import numpy as np 
import pandas as pd 
import time
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from os import walk


path = '/home/khademg/simulationStudy/scripts/'
os.chdir(path)

from classifiers_dataset_info import *

tf.keras.backend.clear_session()

# Select GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

#======================================================================================================
# Read Files and Directories
def directory_contents(path, flag=0):
    f_dirnames=[]
    f_filenames=[]
    for (dirpath, dirnames, filenames) in walk(path):
        f_dirnames.extend(dirnames)
        f_filenames.extend(filenames)
        break
    if flag == 0:
        return sorted(f_dirnames)
    if flag == 1:
        return sorted(f_filenames)

#======================================================================================================
def variational_autoencoder(path, dataset_dict, dataset_name, save_name, n_gen = 1, noise_gen = 0.1,
                            KL_loss_flag = False, model_plot_flag = False, latent_plot_flag = False):
    save_path = '/home/khademg/simulationStudy/scripts/results/Variational_Autoencoder/'
    # Load dataset
    data = pd.read_csv(path)
    n_class = dataset_dict[dataset_name]['nb_classes']
    n_indv_per_class = dataset_dict[dataset_name]['n_indv_per_class']
    n_channels = dataset_dict[dataset_name]['n_channels']
    class_labels = np.array(dataset_dict[dataset_name]['class_labels'])
    ts_l = int(round(len(data.ID)/n_indv_per_class/n_class))  # Length of time series
    sample_size = n_class * n_indv_per_class                       # Sample Size
    X = np.empty((sample_size, ts_l, n_channels))
    y = np.empty((sample_size), dtype=int)
    ID = ["" for x in range(sample_size)]
    original_class = ["" for x in range(sample_size)]
    #----------------------------------
    # Normalize dataset
    mean = np.mean(data.SBP, axis=0)
    std = np.std(data.SBP, axis=0)
    data.SBP -= mean
    data.SBP /= std
    for i in range(sample_size):
        X[i,:,0] = data.SBP[i*ts_l: (i+1)*ts_l]
        y[i] = np.where(class_labels==data.Class[i*ts_l])[0][0]
        ID[i] = data.ID[i*ts_l]
        original_class[i] = data.Class[i*ts_l]
    ID_Class = np.column_stack((ID, original_class))
    slice_ratio=0.5
    indexes = np.arange(len(X))
    np.random.shuffle(indexes)
    n_sample_train = int(np.round(len(X)*slice_ratio))
    train_X = X[indexes[:n_sample_train]]
    test_X = X[indexes[n_sample_train:]]
    #----------------------------------
    # Network parameters
    batch_size = 50
    nb_epochs = 100
    latent_dim = 2
    intermediate_dim = 32
    input_shape = X.shape[1:]
    original_dim = ts_l
    #----------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ Variational Autoencoder (VAE) -----------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    # VAE Reconstruction loss
    def vae_loss(input_layer, output_layer):
        reconstruction_loss = mse(input_layer, output_layer) * original_dim
        return reconstruction_loss
    # KL divergence loss
    class KLDivergenceLayer(layers.Layer):
        """ Identity transform layer that adds KL divergence to the final model loss.    """
        def __init__(self, *args, **kwargs):
            self.is_placeholder = True
            super(KLDivergenceLayer, self).__init__(*args, **kwargs)
        def call(self, inputs):
            z_mean, z_sigma = inputs
            kl_loss = K.sum(1 + z_sigma - K.square(z_mean) - K.exp(z_sigma), axis=-1)
            kl_loss *= -0.5 * 0.1
            self.add_loss(K.mean(kl_loss), inputs=inputs)
            return inputs
    # Latent space sampling function from an isotropic unit Gaussian
    def sample_z(args):
        z_mean, z_sigma = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_sigma) * epsilon
    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- Encoder ----------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    input_layer = Input(shape=input_shape, name='encoder_input')
    x = layers.Conv1D(32, 5, 1, padding='same')(input_layer)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)  #************Dimension Reduction*************#
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(64, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    x = layers.Dense(intermediate_dim, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='latent_mean')(x)
    z_sigma = layers.Dense(latent_dim, name='latent_variance')(x)
    if KL_loss_flag == True:
        z_mean, z_sigma = KLDivergenceLayer()([z_mean, z_sigma])
        model_name = 'vae_cnn_w_KLDivergence_loss_'
    else:
        model_name = 'vae_cnn_wo_KLDivergence_loss_'
    z = layers.Lambda(sample_z, output_shape=(latent_dim, ), name='z')([z_mean, z_sigma])
    # Instantiate encoder model
    encoder = models.Model(input_layer, [z_mean, z_sigma, z], name='encoder')
    if model_plot_flag == True:
        encoder.summary()
        plot_model(encoder, to_file= save_path + model_name + 'encoder.png', show_shapes=True)
    # ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- Decoder ----------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    decoder_input = Input(shape=(latent_dim, ), name='decoder_input')
    x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
    x = layers.Reshape(shape_before_flattening[1:])(x)
    x = layers.Conv1D(64, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.UpSampling1D(2)(x)                          #***********Dimension Upsampling************#
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    decoder_output = layers.Conv1D(n_channels, 5, 1, padding='same', name='decoder_output')(x)
    # Instantiate decoder model
    decoder = models.Model(decoder_input, decoder_output, name='decoder')
    if model_plot_flag == True:
        decoder.summary()
        plot_model(decoder, to_file=save_path + model_name + 'decoder.png', show_shapes=True)
    # ----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- VAE Model -----------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    # Instantiate VAE
    output_layer = decoder(encoder(input_layer)[2])
    vae = models.Model(input_layer, output_layer, name='vae_cnn')
    if model_plot_flag == True:
        vae.summary()
        plot_model(vae, to_file=save_path + model_name + 'encoder_decoder.png', show_shapes=True)
    # Compile VAE
    vae.compile(optimizer='rmsprop', loss=vae_loss)
    # Train autoencoder
    vae.fit(X, X, epochs = nb_epochs, batch_size = batch_size, shuffle=True) # validation_data=(test_X, test_X)
    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- Visualization -----------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    # Display a 2D plot of the classes in the latent space
    if latent_plot_flag == True:
        plt.clf()
        mu, _, _ = encoder.predict(X)
        plt.scatter(mu[:, 0], mu[:, 1], c = y)
        plt.xlabel('z_mean[0]')
        plt.ylabel('z_mean[1]')
        plt.colorbar()
        plt.savefig(save_path + save_name + '_latent_space_mu.pdf', bbox_inches='tight')
        # plt.show()
    df_all = pd.DataFrame(columns=['ID', 'SBP_vae', 'Time', 'Class', 'Gen_ID'])
    for i in range(len(X)):
        # idx2 = data[ data['Class']==ID_Class[i,1] ].index.values
        t = data.Time[0:ts_l].to_numpy()
        ids = [ID_Class[i, 0] for x in range(len(t))]
        clas = [ID_Class[i, 1] for x in range(len(t))]
        z_mean, z_sigma, _ = encoder.predict(X[i:i+1])
        for j in range(n_gen):
            if j == 0:
                z = z_mean
            else:
                epsilon = np.random.normal(0, noise_gen, z_mean.shape)
                z = z_mean + epsilon    # np.exp(0.5*z_sigma[0])*
            z_decoded = decoder.predict(z)
            # z_decoded = np.reshape(z_decoded, (np.prod(z_decoded.shape), 1))
            SBP_vae = z_decoded[0,:,0] * std + mean
            gen_id = ['vae_g'+str(j) for x in range(len(t))]
            data_frame = {'ID':ids, 'SBP_vae':SBP_vae, 'Time': t, 'Class': clas, 'Gen_ID':gen_id}
            df = pd.DataFrame(data_frame)
            df_all = pd.concat((df_all, df), axis=0, sort=False)
    df_all.to_csv(save_path + save_name +'_VAE_generated_dataset_noise_level_' + str(noise_gen) + '.csv', index = False)
    tf.keras.backend.clear_session()

#======================================================================================================

dataset_names = ['mag_cohorts', 'shape_cohorts', 'mag_cohorts_NEW', 'shape_cohorts_NEW']
dataset_names = dataset_names[2:]

for dataset_name in dataset_names:
        nb_classes = dataset_dict[dataset_name]['nb_classes']
        root_path = dataset_dict[dataset_name]['path']
        directory_names = directory_contents(root_path)
        for dirs in directory_names:
            run_path = root_path + dirs
            fnames = directory_contents(run_path, 1)
            for fname in fnames:
                data_path = run_path + '/' + fname
                save_name = dataset_name + '_' + dirs + fname[17:-4]
                variational_autoencoder(data_path, dataset_dict, dataset_name, save_name, n_gen = 4, noise_gen = 1.0,
                                        KL_loss_flag = True, model_plot_flag = False, latent_plot_flag = False)

#================================================================
#================================================================
#================================================================
#================================================================
#================================================================
###                                                    Chollet's Approach                                                      ###

## Define loss
#def vae_loss(input_layer, output_layer):
#  # Reconstruction loss
#  reconstruction_loss = mse(input_layer, output_layer) * original_dim
#  # KL divergence loss
#  kl_loss = K.sum(1 + z_sigma - K.square(z_mean) - K.exp(z_sigma), axis=-1)
#  kl_loss *= -0.5
#  # Total loss = 50% rec + 50% KL divergence loss
#  return K.mean(reconstruction_loss + kl_loss)

### OR

## Define loss
#class CustomVariationalLayer(layers.Layer):
#    def vae_loss(self, input_layer, output_layer, z_mean, z_sigma):
#        input_layer = K.flatten(input_layer)
#        output_layer = K.flatten(output_layer)
#        reconstruction_loss =mse(input_layer, output_layer) * original_dim
#        kl_loss = K.sum(1 + z_sigma - K.square(z_mean) - K.exp(z_sigma), axis=-1)
#        kl_loss *= -0.5
#        return K.mean(reconstruction_loss + kl_loss)
#    def call(self, inputs):
#        input_layer = inputs[0]
#        output_layer = inputs[1]
#        z_mean = inputs[2]
#        z_sigma = inputs[3]
#        loss = self.vae_loss(input_layer, output_layer, z_mean, z_sigma)
#        self.add_loss(loss, inputs=inputs)
#        # We don't use this output.
#        return output_layer

## We call our custom layer on the input and the decoded output,
## to obtain the final model output.
#y = CustomVariationalLayer()([input_layer, output_layer, z_mean, z_sigma])

#vae = models.Model(input_layer, y, name='vae_cnn')
## Compile VAE
#vae.compile(optimizer='adam', loss=None)

#vae.summary()
#plot_model(vae, to_file='vae_cnn_encoder_decoder.png', show_shapes=True)

#vae.fit(x=X, y=None, shuffle=True, epochs=nb_epochs, batch_size=batch_size,)
#================================================================
