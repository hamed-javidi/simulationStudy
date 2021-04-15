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

def CNN_MLP_Autoencoder(save_path, filename, x_train, y_train, x_val, y_val, nb_classes, run, verbose=False,
                        min_exp_val_loss=0.005):
    print("Entered to model")
    np.random.seed()
    batch_size = 50
    nb_epochs = 500
    n_channels = x_train.shape[-1]
    laten_dim = 32
    # Encoder Model
    input_layer = Input(shape=x_train.shape[1:])
    x = layers.Conv1D(32, 5, 1, padding='same')(input_layer)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(64, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    encoder_output = layers.Dense(laten_dim)(x)
    encoder = models.Model(input_layer, encoder_output, name='encoder')
    # Decoder Model
    x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(encoder_output)
    x = layers.Reshape(shape_before_flattening[1:])(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 5, 1, padding='same')(x)
    x = layers.Activation('relu')(x)
    output_layer = layers.Conv1D(n_channels, 5, 1, padding='same', name='decoder_output')(x)

    if input_layer[1].shape != output_layer[1].shape:
        output_layer = layers.Cropping1D(cropping=(1, 1))(output_layer)  # this is the added step

    # Autoencoder Model
    autoencoder = models.Model(input_layer, output_layer, name='autoencoder_cnn')
    autoencoder.compile(loss='mse', optimizer=keras.optimizers.RMSprop())
    autoencoder.fit(x_train, x_train, batch_size=50, epochs=100)
    # Encode Training Data
    x_train = encoder.predict(x_train)
    x_val = encoder.predict(x_val)
    # Train MLP on encoded inputs
    input_layer = Input(shape=x_train.shape[1:])
    x = layers.Flatten()(input_layer)
    x = layers.Dense(100, activation='relu')(x)
    output_layer = layers.Dense(nb_classes, activation='softmax')(x)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['acc'])
    if (verbose == True):
        model.summary()
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=min_exp_val_loss, patience=15)
    baseline_stop = TerminateOnBaseline(monitor='val_acc', baseline=1.0, patience=15)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.92, patience=3, min_lr=0.0001)
    file_path = save_path + 'models/' + filename + '_CNN_MLP_Autoencoder_best_model_run_' + str(run) + '.hdf5'
    model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
    mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
    start_time = time.time()
    hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                     validation_data=(x_val, y_val), verbose=2,
                     callbacks=[early_stop, baseline_stop, reduce_lr, model_checkpoint])
    duration = time.time() - start_time
    print("Elapsed Training Time: %f" % (duration))
    model = models.load_model(file_path)
    y_val_pred = model.predict(x_val)
    y_train_pred = model.predict(x_train)
    training_itrs = len(hist.history['loss'])
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(save_path + 'models/' + filename + '_CNN_MLP_Autoencoder_history_run_' + str(run) + '.csv',
                   index=False)
    keras.backend.clear_session()
    return y_train_pred, y_val_pred, duration, training_itrs  # y_train, y_val, hist
