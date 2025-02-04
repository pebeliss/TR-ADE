import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tensorflow.keras.callbacks import Callback

class EpochLogger(Callback):
    def __init__(self, log_interval=10):
        super().__init__()
        self.log_interval = log_interval  # How often to log metrics (in epochs)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_interval == 0 or epoch == 0:  # Log every `log_interval` epochs
            print(f"Epoch {epoch + 1}: {', '.join([f'{k}={v:.4f}' for k, v in logs.items()])}")

class ResetMetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.val_total_loss.reset_states()
        self.model.val_kl_loss.reset_states()
        self.model.val_recon_loss.reset_states()

class Preprocessing:
    def __init__(self, normal_variables, lognormal_variables, na_cols, imputation_strategy='MICE', iqr_scaler=2, seed=12345):
        self.seed = seed
        self.scaler = StandardScaler()
        self.label_encoder = OneHotEncoder(sparse_output=False, drop=None)
        self.log_shift_values = {}
        self.Q1 = {}
        self.Q3 = {}
        self.IQR = {}
        self.iqr_scaler = iqr_scaler
        self.imputation_strategy = imputation_strategy

        self.all_continuous = normal_variables + lognormal_variables
        self.lognormal_variables = lognormal_variables
        self.normal_variables = normal_variables
        self.normal_variables = normal_variables
        self.na_cols = na_cols

        if self.imputation_strategy == 'MICE':
            self.imputer = IterativeImputer(max_iter=10, random_state=self.seed)
        elif self.imputation_strategy == 'median':
            self.imputer_cont = SimpleImputer(strategy='median')
            self.imputer_bin = SimpleImputer(strategy='most_frequent')

        self.eps = 1e-8

    def fit(self, data):
        # Create a copy of the data for fitting
        data_fit = data.copy()
        self.binary_cols = list(set(data.columns).symmetric_difference(self.all_continuous))
        self.na_cols = list(set(data.columns).intersection(self.na_cols))

        # Step 1: Logtransfrom to fit data transformation for lognormal variables
        data_fit = self.transform_logtransformer(data_fit)
        # Step 2: Remove outliers
        data_fit = self.fit_outlier_handler(data_fit)
        # Step 3: Impute missing values
        self.fit_imputer(data_fit)
        data_fit = self.transform_imputer(data_fit)
        # Step 4: Fit scaling for continuous and lognormal variables (zero mean, unit variance)
        data_fit = self.fit_scaler(data_fit)

    def fit_outlier_handler(self, data_fit):
        for var in self.all_continuous:
            self.Q3[var] = data_fit[var].quantile(0.75)
            self.Q1[var] = data_fit[var].quantile(0.25)
            self.IQR[var] = self.Q3[var] - self.Q1[var]
            outliers = (data_fit[var] < (self.Q1[var] - self.iqr_scaler * self.IQR[var])) | (data_fit[var] > (self.Q3[var] + self.iqr_scaler * self.IQR[var]))
            data_fit.loc[outliers, var] = np.nan
            if len(outliers) > 0 and var not in self.na_cols:
                self.na_cols.append(var)
        return data_fit

    def fit_scaler(self, data_fit):
        if len(self.all_continuous) > 0:
            data_fit[self.all_continuous] = self.scaler.fit_transform(data_fit[self.all_continuous])
        return data_fit

    def fit_imputer(self, data_fit):
        if len(self.na_cols) > 0:
            self.na_cont = list(set(self.na_cols).intersection(self.all_continuous))
            self.na_bin = list(set(self.na_cols).intersection(self.binary_cols))
            if self.imputation_strategy == 'MICE':
                self.imputer.fit(data_fit[self.na_cols])
            elif self.imputation_strategy == 'median':
                if len(self.na_cont) > 0:
                    self.imputer_cont.fit(data_fit[self.na_cont])
                if len(self.na_bin) > 0:
                    self.imputer_bin.fit(data_fit[self.na_bin])

    def transform(self, data):
        # Create a copy of the data to apply transformations
        transformed_data = data.copy()
        # Step 1: Apply the saved logarithmic transformation for lognormal variables
        transformed_data = self.transform_logtransformer(transformed_data)
        # Step 2: Remove outliers in continuous and lognormal variables
        transformed_data = self.transform_outlier_handler(transformed_data)
        # Step 3: Impute
        transformed_data = self.transform_imputer(transformed_data)
        # Step 4: Scale
        transformed_data = self.transform_scaler(transformed_data)
        transformed_data = pd.DataFrame(transformed_data, columns=data.columns, index=data.index)
        return transformed_data

    def transform_logtransformer(self, transformed_data):
        if len(self.lognormal_variables) > 0:
            for var in self.lognormal_variables:
                transformed_data[var] = np.log(transformed_data[var] + self.eps)
        return transformed_data

    def transform_outlier_handler(self, transformed_data):
        if len(self.all_continuous) > 0:
            for var in self.all_continuous:
                outliers = (transformed_data[var] < (self.Q1[var] - self.iqr_scaler * self.IQR[var])) | (transformed_data[var] > (self.Q3[var] + self.iqr_scaler * self.IQR[var]))
                transformed_data.loc[outliers, var] = np.nan  # Mark outliers as NaN without touching missing values
        return transformed_data

    def transform_scaler(self, transformed_data):
        if len(self.all_continuous) > 0:
            transformed_data[self.all_continuous] = self.scaler.transform(transformed_data[self.all_continuous])
        return transformed_data

    def transform_imputer(self, transformed_data):
        if len(self.na_cols) > 0:
            if self.imputation_strategy == 'zero':
                transformed_data.fillna(0, inplace=True)
            elif self.imputation_strategy == 'MICE':
                transformed_data[self.na_cols] = self.imputer.transform(transformed_data[self.na_cols])
                # MICE gives binary variables as success probabilities
                if len(self.na_bin) > 0:
                    transformed_data[self.na_bin] = np.round(transformed_data[self.na_bin])
            elif self.imputation_strategy == 'median':
                if len(self.na_cont) > 0:
                    transformed_data[self.na_cont] = self.imputer_cont.transform(transformed_data[self.na_cont])
                if len(self.na_bin) > 0:
                    transformed_data[self.na_bin] = self.imputer_bin.transform(transformed_data[self.na_bin])
        return transformed_data

    def inverse_transform(self, data):
        reversed_data = data.copy()

        # Reverse scaling of continuous and lognormal variables
        reversed_data = self.inverse_scale(reversed_data)
        # Reverse log transformation of lognormal variables
        reversed_data = self.inverse_logtransformer(reversed_data)
        if len(self.binary_cols) > 0:
            reversed_data[self.binary_cols] = np.round(reversed_data[self.binary_cols])

        return reversed_data

    def inverse_scale(self, data):
        if len(self.all_continuous) > 0:
            data[self.all_continuous] = self.scaler.inverse_transform(data[self.all_continuous])
        return data
    
    def inverse_logtransformer(self, data):
        if len(self.lognormal_variables) > 0:
            for var in self.lognormal_variables:
                data[var] = np.exp(data[var]) + self.eps
        return data

    def encode_labels(self, labels, train=False):
        if train:
            self.label_encoder.fit_transform(labels)
        return self.label_encoder.transform(labels)
    
    def inverse_encode_labels(self, onehot_labels):
        return self.label_encoder.inverse_transform(onehot_labels)
    
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, missing_mask, y=None, batch_size=1000, shuffle=True, **kwargs):
        super(**kwargs).__init__()
        self.X = X.values if isinstance(X, pd.DataFrame) else X  
        self.missing_mask = missing_mask.values if isinstance(missing_mask, pd.DataFrame) else missing_mask  
        if y is not None:
            if isinstance(y, (list, tuple)):  # When y has multiple labels
                self.y = [yi.values if isinstance(yi, pd.DataFrame) else yi for yi in y] 
            else:
                self.y = y.values if isinstance(y, pd.DataFrame) else y
        else:
            self.y = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index, predict=False):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = tf.convert_to_tensor(self.X[batch_indices], dtype=tf.float32)
        if predict:
            return X_batch, batch_indices
        missing_mask_batch = tf.convert_to_tensor(self.missing_mask[batch_indices], dtype=tf.float32)
        if self.y is not None:
            if isinstance(self.y, (list, tuple)):
                y_batch = tuple([tf.convert_to_tensor(yi[batch_indices], dtype=tf.float32) for yi in self.y])
                # y_batch = tuple([yi[batch_indices] for yi in self.y])
            else:
                y_batch = self.y[batch_indices]
            return X_batch, missing_mask_batch, y_batch
        else:
            return X_batch, missing_mask_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)