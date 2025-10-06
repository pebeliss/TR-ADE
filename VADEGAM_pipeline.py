import sys
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

from VADEGAM import VADEGAM
from subclasses import Preprocessing, DataGenerator, EpochLogger

import utils

def set_seed(seed):
        random.seed(seed)  
        np.random.seed(seed) 
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1' 

class PREPROCESSOR_WRAPPER:
    '''Wrapper for preprocessing pipeline, that includes:
        1) Automatic likelihood detection (optional)
        2) Transforming lognormal variables to Gaussian
        3) Outlier handling (cont. variables), imputation and scaling (cont. vars)
        4) Label one-hot encoding (if softmax activation)
        5) Inverting transformations & scaling
    '''
    def __init__(self, args, seed=12345, impute_bf_scaling = True):
        self.seed = args.seed if args.seed is not None else seed
        # set_seed(self.seed)
        # Dataset params
        self.validation_frac = args.validation_frac
        self.iqr_scaler = args.iqr_scaler
        self.imputation_strategy = args.imputation_strategy
        self.scaling_strategy = args.scaling_strategy
        # Model params
        self.classify = args.classify
        self.num_output_head = args.num_output_head
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.final_activation = args.final_activation
        self.impute_bf_scaling = impute_bf_scaling

    def build(self, X_train, X_test=None, data_types = None):
        if data_types is None:
            self.data_types = utils.get_data_types(X_train)
        else:
            self.data_types = data_types
        self.__init_feature_lists__()

        self.na_cols = self.get_na_cols(X_train, X_test)

        self.preprocessor = Preprocessing(self.real_features, self.lognormal_features, self.na_cols, 
                                          imputation_strategy=self.imputation_strategy, iqr_scaler=self.iqr_scaler,
                                          seed=self.seed, impute_bf_scaling=self.impute_bf_scaling, 
                                          scaling_strategy=self.scaling_strategy)
        
    def preprocess(self, X_all, y=None, train=False):
        X = X_all[self.all_features]
        missing_mask = self.get_missing_mask(X)
        missing_mask = missing_mask[self.cont_features + self.bin_features]
        if train:
            self.preprocessor.fit(X)
            if self.classify:
                if self.final_activation == 'softmax':
                    y_onehot = self.preprocessor.encode_labels(y, train=True)
        X_processed = self.preprocessor.transform(X)
        X_processed = X_processed[self.cont_features + self.bin_features]

        if self.classify:
            if self.final_activation == 'softmax':
                y_onehot = self.preprocessor.encode_labels(y)
                y_onehot = pd.DataFrame(y_onehot, columns=self.preprocessor.label_encoder.get_feature_names_out())
                if self.num_output_head > 1:
                    # print(y_onehot)
                    y_onehot = np.array_split(y_onehot, self.num_output_head, axis=1)
                return X_processed, missing_mask, y_onehot
            else:
                if self.num_output_head > 1:
                    y_out = np.array_split(y, self.num_output_head, axis=1)
                return X_processed, missing_mask, y_out

        return X_processed, missing_mask, None
    
    def inverse_preprocess(self, X_reconstruction, y_onehot=None):
        y_labels = None
        X_inverted = self.preprocessor.inverse_transform(X_reconstruction)
        if y_onehot is not None:
            y_labels = self.preprocessor.inverse_encode_labels(y_onehot)
        return X_inverted, y_labels
    
    def get_data_generator(self, X_processed, missing_mask, y_onehot=None, validation=False):
        batch_size = self.batch_size
        if validation:
            batch_size = len(X_processed)
        return DataGenerator(X_processed, missing_mask, y_onehot, batch_size=batch_size)
    
    def data_pipeline(self, X, y=None, return_generator=False, split_seed=None):
        if split_seed is None:
            split_seed = self.seed
        if y is not None:
            X_train, X_val, y_train, y_val = self.train_validation_split(X, y, split_seed=split_seed)
            train_data = self.preprocess(X_train, y_train, train=True)
            val_data = self.preprocess(X_val, y_val)
        else:
            X_train, X_val = self.train_validation_split(X, split_seed=split_seed)
            train_data = self.preprocess(X_train, train=True)
            val_data = self.preprocess(X_val)
        if return_generator:
            train_generator = self.get_data_generator(*train_data)
            val_generator = self.get_data_generator(*val_data, validation=True)
            return train_generator, val_generator, train_data, val_data
        else:
            return train_data, val_data
        
    def train_validation_split(self, X, y=None, split_seed=None):
        if split_seed is None:
            split_seed = self.seed
        if y is None:
            return train_test_split(X, test_size=self.validation_frac, random_state=split_seed)
        return train_test_split(X, y, test_size=self.validation_frac, stratify=y, random_state=split_seed)

    def __init_feature_lists__(self):
        self.real_features = list(self.data_types[self.data_types['type'] == 'norm'].index)
        self.lognormal_features = list(self.data_types[self.data_types['type'] == 'lognorm'].index)
        self.bin_features = list(self.data_types[self.data_types['type'] == 'bin'].index)
        self.cont_features = self.real_features + self.lognormal_features
        self.all_features = self.cont_features + self.bin_features
        # Dims
        self.cont_dim, self.bin_dim = len(self.cont_features), len(self.bin_features)

    def get_missing_mask(self, data):
        return data.isnull().astype(int).astype(float)

    def get_na_cols(self, X_train, X_test):
        na_cols = set()
        for df in [X_train, X_test]:
            if df is not None:
                na_cols_subset = df.columns[df.isnull().any()]  # Columns with at least one NaN
                na_cols.update(na_cols_subset)
        return list(na_cols)

class VADEGAM_WRAPPER:
    '''Wrapper for VaDEGam model with customable lr schedule & early stopping.'''
    def __init__(self, args, cont_dim, bin_dim, seed=12345, log_interval=50, classif_dependent=True):
        # set_seed(seed)
        self.seed = args.seed if args.seed is not None else seed
        self.args = args
        self.log_interval = log_interval
        self.cont_dim = cont_dim 
        self.bin_dim = bin_dim
        self.classif_dependent = classif_dependent
        self.set_args(self.args)

    def set_args(self, args):
        # Paths
        self.data_path = args.data_path
        self.results_path = args.results_path
        # Optimization params
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.use_early_stopping = args.use_early_stopping
        self.early_stop_patience = args.early_stopping_patience
        self.learning_rate = args.learning_rate
        self.use_lr_schedule = args.use_lr_schedule
        self.callbacks = []
        # Model params
        self.classify = args.classify
        self.num_clusters = args.num_clusters
        self.latent_dim = args.latent_dim
        self.gamma = args.gamma
        self.s_to_classifier = args.s_to_classifier
        self.learn_prior = args.learn_prior
        self.num_output_head = args.num_output_head
        self.num_classes = args.num_classes
        self.final_activation = args.final_activation
        self.nn_layers = args.nn_layers
        self.c_sigma_initializer = args.c_sigma_initializer

    def build(self):
        self.build_vadegam()

        if self.use_early_stopping:
            self.early_stopping = tf.keras.callbacks.EarlyStopping(
                                monitor="val_total_loss",
                                patience=self.early_stop_patience,
                                min_delta=0.001,
                                restore_best_weights=True,
                                mode='min')
            self.callbacks.append(self.early_stopping)
        if self.use_lr_schedule:
            self.lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_total_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6)
            self.callbacks.append(self.lr_schedule)

        epoch_logger = EpochLogger(log_interval=self.log_interval)
        self.callbacks.append(epoch_logger)
            
    def build_vadegam(self):
        self.vadegam = VADEGAM(self.latent_dim, self.cont_dim, self.bin_dim, 
                               self.num_classes, self.gamma, classify=self.classify, 
                               num_output_head=self.num_output_head, num_clusters=self.num_clusters, 
                               s_to_classifier=self.s_to_classifier, learn_prior=self.learn_prior,
                               final_activation=self.final_activation, nn_layers=self.nn_layers, initializer=self.c_sigma_initializer,
                               classif_dependent=self.classif_dependent)
        if self.use_lr_schedule:
            self.vadegam.compile(optimizer=keras.optimizers.Adam())
        else:
            self.vadegam.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))

    def model_fit(self, train_generator, val_generator):
        if self.use_early_stopping:
            return self.vadegam.fit(train_generator, validation_data=val_generator, 
                                    epochs=self.epochs, callbacks=self.callbacks, verbose=0)
        return self.vadegam.fit(train_generator, validation_data=val_generator, 
                                epochs=self.epochs, callbacks=self.callbacks, verbose=0)

    def model_predict(self, X_processed):
        out = self.vadegam.predict(X_processed)
        X_recon, z_mean, clusters = out[:3]
        if self.classify:
            pred_label = np.concatenate(out[3:], axis=1)
            return X_recon, z_mean, clusters, pred_label
        return X_recon, z_mean, clusters
    
    def get_clusters(self, z_mean):
        return self.vadegam.get_clusters(z_mean).numpy()
    