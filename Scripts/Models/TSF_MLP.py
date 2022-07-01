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
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from math import sqrt
import keras_tuner as kt

# ======================================================================================================
input_shape = 0
n_classes = 2


class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Flatten(input_shape=input_shape))
        model.add(layers.Dropout(hp.Float('dropout0', min_value=0, max_value=0.8, step=0.1, default=0.1)))
        model.add(layers.Dense(hp.Choice('dense1', [32, 64, 128, 512], default=512), activation='relu'))
        model.add(layers.Dropout(hp.Float('dropout1', min_value=0, max_value=0.8, step=0.1, default=0.1)))
        model.add(layers.Dense(hp.Choice('dense2', [32, 64, 128, 512], default=512), activation='relu'))
        model.add(layers.Dropout(hp.Float('dropout2', min_value=0, max_value=0.8, step=0.1, default=0.2)))
        model.add(layers.Dense(hp.Choice('dense3', [32, 64, 128, 512], default=512), activation='relu'))
        model.add(layers.Dropout(hp.Float('dropout3', min_value=0, max_value=0.8, step=0.1, default=0.3)))
        model.add(layers.Dense(n_classes, activation='softmax'))
        # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batchsize', [64, 128, 256, 512], default=64),
            **kwargs,
        )


def TSF_MLP_classifier(X, Y, x_test, y_test, params):
    keras.backend.clear_session()
    np.random.seed(params['seed'])
    model_name = 'TSF_MLP'
    nb_epochs = 500

    global input_shape
    global n_classes
    n_classes = params["nb_classes"]
    X, x_test = TSF_dataset(X, x_test)
    input_shape = X.shape[1:]
    tuner, best_hps = tune_model(MyHyperModel, params["save_path"], X, Y, params["seed"], model_name + '_' + params["tune_project_name"], params)
    df_metrics = []
    rskf = RepeatedStratifiedKFold(n_splits=params["folds"], n_repeats=1, random_state=params['seed'])
    fold = 0
    for train_index, val_index in rskf.split(X, np.argmax(Y, axis=1)):
        attempt = 0  # Train the model up to 3 attempts to make sure training was succesfull
        attempt_stop = False
        while not attempt_stop:
            attempt += 1
            model = tuner.hypermodel.build(best_hps)
            fold += 1
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]
            early_stop = callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=15)
            baseline_stop = TerminateOnBaseline(monitor='val_acc', baseline=1.0, patience=15)
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.92, patience=5)
            file_path = params["save_path"] + 'models/' + params["save_name"] + '_' + model_name + '_fold_' + str(fold) + '.hdf5'
            model_checkpoint = callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
            mini_batch_size = int(min(x_train.shape[0], best_hps.get('batchsize')))
            start_time = time.time()
            hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                             validation_data=(x_val, y_val), verbose=2,
                             callbacks=[
                                 early_stop, reduce_lr,
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
            if (val_AUC > 0.7) and (test_AUC > 0.7):
                attempt_stop = True
            elif (attempt >= 3) and (val_AUC > 0.5) and (test_AUC > 0.5):
                attempt_stop = True
            elif attempt >= 5:
                attempt_stop = True
            else:
                continue
            best_thrsh = GetBestThreshold(y_test, y_test_pred)
            # best_thresholds.append(best_thrsh)
            accuracy = accuracy_score(np.argmax(y_test, axis=1), (y_test_pred[:, 1] >= best_thrsh).astype("int"))
            # acc_bests.append(accuracy)
            # train_aucs.append(train_AUC)
            # val_aucs.append(val_AUC)
            # test_aucs.append(test_AUC)
            training_itrs = len(hist.history['loss'])
            # mean_itrs.append(training_itrs)
            # mean_durs.append(duration)
            hist_df = pd.DataFrame(hist.history)
            hist_df.to_csv(params["save_path"] + 'models/' + params["save_name"] + '_' + model_name + '_history_fold' + str(fold) + '.csv', index=False)
            metrics = calculate_metrics(y_test, y_test_pred, params["save_name"], train_AUC, val_AUC, test_AUC, best_thrsh, accuracy, duration, training_itrs)
            metrics = metrics.squeeze()
            df_metrics.append(pd.DataFrame(metrics.append(pd.Series(best_hps.values))).transpose())
    # mean_test_auc = np.mean(test_aucs)
    # std_test_auc = np.std(test_aucs)
    # mean_val_auc = np.mean(val_aucs)
    # std_val_auc = np.std(val_aucs)
    # mean_train_auc = np.mean(train_aucs)
    # std_train_auc = np.std(train_aucs)
    df_metrics = pd.concat(df_metrics, axis=0, sort=False)
    return y_train_pred, y_val_pred, y_test_pred, df_metrics

"""    
    # Data Preparation for Time Series Forest (TSF) Method
    x_train, x_val, x_test = TSF_dataset(x_train, x_val, x_test)
    input_layer = Input(shape=x_train.shape[1:])
    # flatten/reshape because when multivariate all should be on the same axis
    input_layer_flattened = layers.Flatten()(input_layer)
    x = layers.Dropout(0.1)(input_layer_flattened)
    x = layers.Dense(500, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(500, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(500, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output_layer = layers.Dense(nb_classes, activation='softmax')(x)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['acc'])
    if (verbose == True):
        model.summary()
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=min_exp_val_loss, patience=15)
    baseline_stop = TerminateOnBaseline(monitor='val_acc', baseline=1.0, patience=15)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.92, patience=5, min_lr=0.0001)
    file_path = save_path + 'models/' + filename + '_TSF_MLP_best_model_run_' + str(run) + '.hdf5'
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
    y_test_pred = model.predict(x_test)
    training_itrs = len(hist.history['loss'])
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(save_path + 'models/' + filename + '_TSF_MLP_history_run_' + str(run) + '.csv', index=False)
    keras.backend.clear_session()

    if verbose == 1:
        val_perf = mean_absolute_error(y_val, y_val_pred), sqrt(mean_squared_error(y_val, y_val_pred))
        test_perf = mean_absolute_error(y_test, y_test_pred), sqrt(mean_squared_error(y_test, y_test_pred))
        print(f'MAE= {abs(val_perf[0] - test_perf[0]) < 0.01}, val: {val_perf[0]}, test: {test_perf[0]}')
        print(f'RMSE= {abs(val_perf[1] - test_perf[1]) < 0.01}, val: {val_perf[1]}, test: {test_perf[1]}')

    return y_train_pred, y_val_pred, y_test_pred, duration, training_itrs
"""

def TSF_dataset(x_train, x_test):
    n_train = x_train.shape[0]
    # n_val = n_train + x_val.shape[0]
    X = np.concatenate((x_train, x_test), axis=0)
    n_samples = X.shape[0]
    m = X.shape[1]  # Timeseries length
    n_channels = X.shape[2]
    p = m // 5  # Minimum sub-series length
    if (p < 2) & (m >= 4):
        p = 2
    elif (p < 2) & (m < 4):
        print("Unexpected error in TSF-CNN")
        print(f"m: {m}, p: {p}")
        raise

    n_interval = int(np.round(np.sqrt(m)))  # Number of intervals
    X_TSF = np.empty((n_samples, n_interval * 3, n_channels))
    for i in range(n_channels):
        for j in range(n_interval):
            a = np.random.randint(0, m - p + 1)
            b = np.random.randint(a + p, m + 1)
            for k in range(n_samples):
                # Apply on Train dataset
                X_TSF[k, 3 * j + 0, i] = np.mean(X[k, a:b, i])
                X_TSF[k, 3 * j + 1, i] = np.std(X[k, a:b, i])
                X_TSF[k, 3 * j + 2, i] = np.polyfit(np.arange(b - a), X[k, a:b, i], 1)[0]
    x_train = X_TSF[:n_train]
    x_test = X_TSF[n_train:]
    return x_train, x_test