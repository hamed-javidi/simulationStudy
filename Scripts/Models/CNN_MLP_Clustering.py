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


# ======================================================================================================

def CNN_MLP_Clustering(save_path, filename, x_train, nb_classes, run, file_path, verbose=False, nb_epochs=500):
    np.random.seed()
    batch_size = 50
    n_channels = x_train.shape[-1]
    laten_dim = 16
    intermediate_dim = 32
    # Encoder Model
    print("1 - Input shape : " + str(x_train.shape[1:]))
    input_layer = Input(shape=x_train.shape[1:])
    # if x_train.shape[1] % 2 == 1:
    #     input_layer = layers.ZeroPadding1D(padding=(1,0))(input_layer)

    x = layers.Conv1D(32, 5, 1, padding='same')(input_layer)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    # x = layers.MaxPooling1D(1, padding='same')(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(64, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    x = layers.Dense(intermediate_dim, activation='relu')(x)
    encoder_output = layers.Dense(laten_dim)(x)
    # Decoder Model
    x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(encoder_output)
    x = layers.Reshape(shape_before_flattening[1:])(x)
    x = layers.Conv1D(64, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.UpSampling1D(2)(x)
    # x = layers.UpSampling1D(1)(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    output_layer = layers.Conv1D(n_channels, 5, 1, padding='same', name='decoder_output')(x)

    if input_layer[1].shape != output_layer[1].shape:
        output_layer = layers.Cropping1D(cropping=(0, 1))(output_layer)  # this is the added step
    print("1 - Output layer shape : " + str(output_layer.shape))

    # Autoencoder Model
    autoencoder = models.Model(input_layer, output_layer, name='autoencoder_cnn')
    autoencoder.compile(loss='mse', optimizer=keras.optimizers.RMSprop())

    early_stop = callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=15)
    model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

    start_time = time.time()
    hist = autoencoder.fit(x_train, x_train, batch_size=batch_size, epochs=nb_epochs, verbose=2,
                         callbacks=[early_stop, model_checkpoint])
    duration = time.time() - start_time
    training_itrs = len(hist.history['loss'])
    #print("Elapsed Training Time: %f" % (duration))
    # encoder = models.Model(inputs = autoencoder.inputs, outputs = autoencoder.layers[12].output)

    # Save Model
    # file_path = save_path + 'models/' + filename + '_CNN_MLP_Clustering_run_' + str(run) + '.hdf5'
    autoencoder.save(file_path)
    losses = hist.history['loss']
    keras.backend.clear_session()
    return losses[-1], duration, training_itrs
