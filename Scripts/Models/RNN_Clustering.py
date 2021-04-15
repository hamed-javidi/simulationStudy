# Fully CNN Classifier
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
from manual_early_stop import TerminateOnBaseline
from tensorflow.keras import backend as K
tf.keras.backend.clear_session()
#======================================================================================================

def RNN_Clustering(save_path, filename, x_train, nb_classes, run, file_path, verbose=False, nb_epochs=500):
        np.random.seed()
        batch_size = 50
        n_channels = x_train.shape[-1]
        timesteps = x_train.shape[1]
        laten_dim = 2
        # Encoder Model
        input_layer = Input(shape = x_train.shape[1:])
        x = layers.LSTM(128, return_sequences=True)(input_layer)
        x = layers.LSTM(64,   return_sequences=True)(x)
        encoded = layers.LSTM(laten_dim)(x)
        # Decoder Model
        x = layers.RepeatVector(timesteps)(encoded)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        output_layer = layers.TimeDistributed(layers.Dense(n_channels))(x)
        # Autoencoder Model
        autoencoder = models.Model(input_layer, output_layer, name='autoencoder_cnn')
        autoencoder.compile(loss='mse', optimizer = keras.optimizers.RMSprop())
        # Save Model
        #file_path = save_path + 'models/' + filename + '_RNN_Clustering_run_' + str(run) + '.hdf5'

        early_stop = callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=15)
        model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        start_time = time.time()
        hist = autoencoder.fit(x_train, x_train, batch_size = batch_size, epochs = nb_epochs, verbose=2,
                         callbacks=[early_stop, model_checkpoint])
        duration = time.time() - start_time
        training_itrs = len(hist.history['loss'])
        print("Elapsed Training Time: %f" % (duration))
        # encoder = models.Model(inputs = autoencoder.inputs, outputs = autoencoder.layers[12].output)
        autoencoder.save(file_path)
        losses = hist.history['loss']
        keras.backend.clear_session()
        return losses[-1], duration, training_itrs
