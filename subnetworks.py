import tensorflow as tf
from tensorflow import keras
from keras import layers

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_logvar = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon

class Encoder(keras.Model):
    def __init__(self, latent_dim, hidden1 = 64, hidden2=32, hidden3=16, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.enc_layers = [] 
        if hidden1 is not None:
            self.enc_layers.append(layers.Dense(hidden1, activation="relu", name='hidden1'))
        if hidden2 is not None:
            self.enc_layers.append(layers.Dense(hidden2, activation="relu", name='hidden2'))
        if hidden3 is not None:
            self.enc_layers.append(layers.Dense(hidden3, activation="relu", name='hidden3'))
        if len(self.enc_layers) == 0:
            self.enc_layers.append(layers.Dense(32, activation="relu", name='default_fc'))
        
        self.z_mean_layer = layers.Dense(latent_dim, name='z_mean')
        self.z_logvar_layer = layers.Dense(latent_dim, name='z_logvar')

    def call(self, inputs):
        x = inputs
        for layer in self.enc_layers:
            x = layer(x)
        z_mean = self.z_mean_layer(x)
        z_logvar = self.z_logvar_layer(x)
        # z_logvar = tf.clip_by_value(z_logvar, -10.0, 10.0) # NAn fix try
        tf.debugging.check_numerics(z_mean, "DEC: z_mean has NaN or Inf!")
        tf.debugging.check_numerics(z_logvar, "DEC: z_logvar has NaN or Inf!")
        z = Sampling()([z_mean, z_logvar])
        return z_mean, z_logvar, z


class Decoder(keras.Model):
    def __init__(self, dim_cont, dim_binary, hidden1 = 16, hidden2=32, hidden3=64, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dec_layers = [] 
        if hidden1 is not None:
            self.dec_layers.append(layers.Dense(hidden1, activation="relu", name='hidden1'))
        if hidden2 is not None:
            self.dec_layers.append(layers.Dense(hidden2, activation="relu", name='hidden2'))
        if hidden3 is not None:
            self.dec_layers.append(layers.Dense(hidden3, activation="relu", name='hidden3'))
        if len(self.dec_layers) == 0:
            self.dec_layers.append(layers.Dense(32, activation="relu", name='default_fc'))
        
        self.epsilon = tf.constant(1e-3, dtype=tf.float32)
        self.x_mean_pred_layer = layers.Dense(dim_cont, name='x_mean_pred')
        self.x_logvar_pred_layer = layers.Dense(dim_cont, name='x_logvar_pred')
        self.theta_pred_layer = layers.Dense(dim_binary, activation='sigmoid', name='theta_pred')

    def call(self, z):
        x = z
        for layer in self.dec_layers:
            x = layer(x)
        x_mean_pred = self.x_mean_pred_layer(x)
        x_logvar_pred = self.x_logvar_pred_layer(x)
        x_logvar_pred = tf.clip_by_value(tf.nn.softplus(x_logvar_pred), self.epsilon, 1e20) # positive

        theta_pred = self.theta_pred_layer(x)
        # reconstruction = tf.concat([x_mean_pred, theta_pred], axis=1)
        # return reconstruction
        return x_mean_pred, x_logvar_pred, theta_pred


class Classifier(keras.Model):
    def __init__(self, num_classes, hidden1=16, hidden2=None, num_output_head=1, final_activation='softmax', **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.class_layers = []
        self.num_output_head = num_output_head

        if hidden1 is not None:
            self.class_layers.append(layers.Dense(hidden1, activation="relu"))
        if hidden2 is not None:
            self.class_layers.append(layers.Dense(hidden2, activation="relu"))
        if len(self.class_layers) == 0:
            self.class_layers.append(layers.Dense(2, activation="relu"))

        self.pred_y_layer1 = layers.Dense(num_classes, activation=final_activation)
        if self.num_output_head > 1:
            self.pred_y_layer2 = layers.Dense(num_classes, activation=final_activation)
        if self.num_output_head > 2:
            self.pred_y_layer3 = layers.Dense(num_classes, activation=final_activation)

    def call(self, z):
        x = z
        for layer in self.class_layers:
            x = layer(x)
        pred_y1 = self.pred_y_layer1(x)
        if self.num_output_head > 1:
            pred_y2 = self.pred_y_layer2(x)
        else:
            pred_y2 = None
        if self.num_output_head > 2:
            pred_y3 = self.pred_y_layer3(x)
        else:
            pred_y3 = None
        return pred_y1, pred_y2, pred_y3
    
class Classifier_indep(keras.Model):
    def __init__(self, num_classes, hidden1=16, hidden2=None, final_activation='softmax', **kwargs):
        super(Classifier_indep, self).__init__(**kwargs)
        self.class_layers = []

        if hidden1 is not None:
            self.class_layers.append(layers.Dense(hidden1, activation="relu"))
        if hidden2 is not None:
            self.class_layers.append(layers.Dense(hidden2, activation="relu"))
        if len(self.class_layers) == 0:
            self.class_layers.append(layers.Dense(2, activation="relu"))

        self.pred_y_layer = layers.Dense(num_classes, activation=final_activation)

    def call(self, z):
        x = z
        for layer in self.class_layers:
            x = layer(x)
        pred_y = self.pred_y_layer(x)
        return pred_y

    
class Classifier_New(keras.Model):
    def __init__(self, num_classes, num_clusters, num_output_head=1, hidden1=3, final_activation='softmax', **kwargs):
        super(Classifier_New, self).__init__(**kwargs)
        self.class_layers = []
        self.num_output_head = num_output_head
        self.num_clusters = num_clusters
        self.num_classes = num_classes

        # self.feature_extractor_layer = layers.Dense(hidden1, activation="relu")

        # Cluster-specific weights and biases (one per cluster)
        self.W_c1 = [self.add_weight(shape=(hidden1, self.num_classes), 
                                    initializer='random_normal', 
                                    name=f'W1_cluster_{i}') for i in range(self.num_clusters)]
        # self.b_c1 = [self.add_weight(shape=(self.num_classes,), 
        #                             initializer='zeros', 
        #                             name=f'b1_cluster_{i}') for i in range(self.num_clusters)]

        if num_output_head > 1:
            self.W_c2 = [self.add_weight(shape=(hidden1, self.num_classes), 
                                    initializer='random_normal', 
                                    name=f'W2_cluster_{i}') for i in range(self.num_clusters)]
            # self.b_c2 = [self.add_weight(shape=(self.num_classes,), 
            #                         initializer='zeros', 
            #                         name=f'b2_cluster_{i}') for i in range(self.num_clusters)]
            
        if num_output_head > 2:
            self.W_c3 = [self.add_weight(shape=(hidden1, self.num_classes), 
                                    initializer='random_normal', 
                                    name=f'W3_cluster_{i}') for i in range(self.num_clusters)]
            # self.b_c3 = [self.add_weight(shape=(self.num_classes,), 
            #                         initializer='zeros', 
            #                         name=f'b3_cluster_{i}') for i in range(self.num_clusters)]

    def call(self, inputs):
        z, c = inputs
        # x = self.feature_extractor_layer(z)
        x = z
        # Initialize list for the outputs (predictions for each class)
        predictions1 = []
        predictions2 = []
        predictions3 = []


        # For each cluster, use the corresponding W_c and b_c
        for i in range(self.num_clusters):
            # Select cluster-specific weights and bias using the cluster assignment c
            W1 = self.W_c1[i]
            # Compute logits for this cluster
            logits1 = tf.matmul(x, W1)
            # Apply softmax to get class probabilities
            prob1 = tf.nn.softmax(logits1, axis=-1)
            # Store the prediction
            predictions1.append(prob1)

            if self.num_output_head > 1:
                W2 = self.W_c2[i]
                logits2 = tf.matmul(x, W2)
                prob2 = tf.nn.softmax(logits2, axis=-1)
                predictions2.append(prob2)

            if self.num_output_head > 2:
                W3 = self.W_c3[i]
                logits3 = tf.matmul(x, W3)
                prob3 = tf.nn.softmax(logits3, axis=-1)
                predictions3.append(prob3)
        
        c_expanded = tf.expand_dims(c, -1)
        # stacked1 = tf.transpose(tf.stack(predictions1), perm=[1, 0, 2])
        stacked1 = tf.stack(predictions1, axis=1)
        y_pred1 = tf.reduce_sum(stacked1 * c_expanded, axis=1)

        if self.num_output_head > 1:
            stacked2 = tf.stack(predictions2, axis=1)
            y_pred2 = tf.reduce_sum(stacked2 * c_expanded, axis=1)
        else:
            y_pred2 = None
        if self.num_output_head > 2:
            stacked3 = tf.stack(predictions3, axis=1)
            y_pred3 = tf.reduce_sum(stacked3 * c_expanded, axis=1)
        else:
            y_pred3 = None

        return y_pred1, y_pred2, y_pred3
