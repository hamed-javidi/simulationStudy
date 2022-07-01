import time
from TSC_utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from manual_early_stop import TerminateOnBaseline
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

tf.keras.backend.clear_session()
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from math import sqrt
import keras_tuner as kt

# ======================================================================================================
input_shape = 0
n_classes = 2


class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        num_transformer_blocks = 4
        inputs = keras.Input(shape=input_shape)
        x= inputs
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, hp.Choice('headsize', [1, 2, 4, 8], default=2), hp.Choice('n_heads', [1, 2, 4, 8], default=8), hp.Choice('n_heads', [1, 2, 4, 8], default=4), hp.Float('encoder_dropout', min_value=0, max_value=0.8, step=0.1, default=0.1))

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        x = layers.Dense(hp.Choice('dense1', [32, 64, 128, 256], default=128), activation="relu")(x)
        x = layers.Dropout(hp.Float('dropout1', min_value=0, max_value=0.8, step=0.1, default=0.1))(x)
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        model = models.Model(inputs, outputs)
        global_step = tf.Variable(1, name="global_step")
        lr = noam_scheme(0.0003, global_step, 4000)
        optimizer = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batchsize', [64, 128, 256, 512], default=64),
            **kwargs,
        )

# Use the Adam optimizer with a custom learning rate scheduler according to the formula in the paper.
def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


# def build_model(
#         input_shape,
#         head_size,
#         num_heads,
#         ff_dim,
#         num_transformer_blocks,
#         mlp_units,
#         dropout=0,
#         mlp_dropout=0,
# ):
#     n_classes = 2
#     inputs = keras.Input(shape=input_shape)
#     x = inputs
#     for _ in range(num_transformer_blocks):
#         x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
#
#     x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
#     for dim in mlp_units:
#         x = layers.Dense(dim, activation="relu")(x)
#         x = layers.Dropout(mlp_dropout)(x)
#     outputs = layers.Dense(n_classes, activation="softmax")(x)
#     return keras.Model(inputs, outputs)


def Transformer_classifier(X, Y, x_test, y_test, params):
    keras.backend.clear_session()
    np.random.seed(params['seed'])
    model_name = 'Transformer'
    nb_epochs = 500

    global input_shape
    global n_classes
    input_shape = X.shape[1:]
    n_classes = params["nb_classes"]

    tuner, best_hps = tune_model(MyHyperModel, params["save_path"], X, Y, params["seed"], model_name + '_' + params["tune_project_name"], params)

    df_metrics = []
    rskf = RepeatedStratifiedKFold(n_splits=params["folds"], n_repeats=1, random_state=params['seed'])
    fold = 0
    for train_index, val_index in rskf.split(X, np.argmax(Y, axis=1)):
        fold += 1
        attempt = 0  # Train the model up to 3 attempts to make sure training was succesfull
        attempt_stop = False
        while not attempt_stop:
            attempt += 1
            model = tuner.hypermodel.build(best_hps)
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            early_stop = callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=15, min_delta=params['min_exp_val_loss'])
            baseline_stop = TerminateOnBaseline(monitor='val_acc', baseline=1.0, patience=15)
            # reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.92, patience=5)
            file_path = params["save_path"] + 'models/' + params["save_name"] + '_' + model_name + '_fold_' + str(fold) + '.hdf5'
            model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
            # mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
            start_time = time.time()
            hist = model.fit(x_train, y_train, batch_size=best_hps.get('batchsize'), epochs=nb_epochs,
                             validation_data=(x_val, y_val), verbose=2,
                             callbacks=[
                                 early_stop,
                                 baseline_stop,
                                 model_checkpoint]
                             )
            duration = time.time() - start_time
            model = models.load_model(file_path)
            y_train_pred = model.predict(x_train)
            y_val_pred = model.predict(x_val)
            y_test_pred = model.predict(x_test)

            train_AUC = roc_auc_score(np.argmax(Y[train_index], axis=1), y_train_pred[:, 1])
            val_AUC = roc_auc_score(np.argmax(Y[val_index], axis=1), y_val_pred[:, 1])
            test_AUC = roc_auc_score(np.argmax(y_test, axis=1), y_test_pred[:, 1], average='macro')
            if val_AUC > 0.7:
                attempt_stop = True
            elif val_AUC < 0.5:
                continue
            elif attempt >= 3 and val_AUC >= 0.5:
                attempt_stop = True
            elif attempt > 10:
                attempt_stop = True
            else:
                continue
            print(f'Train AUC: {train_AUC}\n'
                  f'Val AUC: {val_AUC}\n'
                  f'Test AUC: {test_AUC}')
            best_thrsh = GetBestThreshold(y_test, y_test_pred)
            # best_thresholds.append(best_thrsh)
            accuracy = accuracy_score(np.argmax(y_test, axis=1), (y_test_pred[:, 1] >= best_thrsh).astype("int"))
            training_itrs = len(hist.history['loss'])
            # mean_itrs.append(training_itrs)
            # mean_durs.append(duration)
            hist_df = pd.DataFrame(hist.history)
            hist_df.to_csv(params["save_path"] + 'models/' + params["save_name"] + '_' + model_name + '_history_fold' + str(fold) + '.csv', index=False)
            metrics = calculate_metrics(y_test, y_test_pred, params["save_name"], train_AUC, val_AUC, test_AUC, best_thrsh, accuracy, duration, training_itrs)
            metrics = metrics.squeeze()
            df_metrics.append(pd.DataFrame(metrics.append(pd.Series(best_hps.values))).transpose())
    df_metrics = pd.concat(df_metrics, axis=0, sort=False)
    return y_train_pred, y_val_pred, y_test_pred, df_metrics

"""
    d_model = 256  # or head size
    h = 8
    model = build_model(
        # # Transformer base model
        # input_shape,
        # head_size=512, # or d_model
        # num_heads=8,
        # ff_dim=2048,
        # num_transformer_blocks=6,
        # mlp_units=[128],
        # mlp_dropout=0.1,
        # dropout=0.1

        # Transformer abstract model
        input_shape,
        head_size=d_model,
        num_heads=h,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.1,
        dropout=0.1,
    )
    global_step = tf.Variable(1, name="global_step")
    lr = noam_scheme(0.0003, global_step, 4000)
    optimizer = tf.keras.optimizers.Adam(lr)

    file_path = save_path + 'models/' + filename + '_Transformmer_best_model_run_' + str(run) + '.hdf5'
    model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=50)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=['mse']
    )
    # model.summary()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val), verbose=1,
        epochs=nb_epochs,
        batch_size=64,
        callbacks=[model_checkpoint, early_stop]
    )
    model = keras.models.load_model(file_path)
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)
    train_AUC = roc_auc_score(np.argmax(y_train, axis=1), y_train_pred[:, 1], average='macro')
    test_AUC = roc_auc_score(np.argmax(y_test, axis=1), y_test_pred[:, 1], average='macro')
    val_AUC = roc_auc_score(np.argmax(y_val, axis=1), y_val_pred[:, 1], average='macro')

    fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), y_test_pred[:, 1], pos_label=1)
    ix = np.argmax(tpr - fpr)
    best_thrsh = thresholds[ix] if thresholds[ix] < 1 else 0.5
    accuracy = accuracy_score(np.argmax(y_test, axis=1), (y_test_pred[:, 1] >= best_thrsh).astype("int"))

"""
