import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from subnetworks import Sampling, Encoder, Decoder, Classifier, Classifier_New, Classifier_indep

class VADEGAM(keras.Model):
    def __init__(self, latent_dim, cont_dim, bin_dim, num_classes=None, gamma = 1, 
                 classify= True, num_output_head=1, num_clusters = 10, learn_prior = False, 
                 s_to_classifier=False, final_activation='softmax', nn_layers = [64, 32, 16], initializer = 'glorot', 
                 classif_dependent=True, **kwargs):
        super(VADEGAM, self, **kwargs).__init__()
        '''Class constructor.'''
        self.classify = classify
        self.s_to_classifier = s_to_classifier
        self.encoder = Encoder(latent_dim, hidden1=nn_layers[0], hidden2=nn_layers[1], hidden3=nn_layers[2])
        self.decoder = Decoder(cont_dim, bin_dim, hidden1=nn_layers[2], hidden2=nn_layers[1], hidden3=nn_layers[0])
        self.classif_dependent = classif_dependent
        
        if self.classify:
            if self.s_to_classifier:
                self.classifier = Classifier_New(num_classes, num_clusters, num_output_head=num_output_head, hidden1=latent_dim)
            else:
                if self.classif_dependent:
                    self.classifier = Classifier(num_classes, num_output_head=num_output_head, final_activation=final_activation)
                else:
                    self.classifier1 = Classifier_indep(num_classes, final_activation=final_activation)
                    self.classifier2 = Classifier_indep(num_classes, final_activation=final_activation)
                    self.classifier3 = Classifier_indep(num_classes, final_activation=final_activation)
                    
        self.gamma = gamma
        self.z_dim = latent_dim
        self.cont_dim = cont_dim
        self.bin_dim = bin_dim
        self.num_output_head = num_output_head
        self.num_classes = num_classes
        self.final_activation = final_activation

        # GMM prior specific params
        self.num_clusters = num_clusters
        self.learn_prior = learn_prior

        self.c_mu = tf.Variable(tf.initializers.GlorotNormal()(shape=[self.num_clusters, self.z_dim]), name='c_mu', trainable=True)
        if initializer == 'glorot':
            self.log_c_sigma = tf.Variable(tf.initializers.GlorotNormal()([self.num_clusters, self.z_dim]), name="c_sigma", trainable=True)
        else:
            self.log_c_sigma = tf.Variable(tf.constant_initializer(value=0.01)([self.num_clusters, self.z_dim]), name="c_sigma", trainable=True)
        
        self._track_variable(self.c_mu)
        self._track_variable(self.log_c_sigma)
        
        if self.learn_prior:
            self.prior_logits = tf.Variable(tf.ones([self.num_clusters]), name="c_prior", trainable=True)
            self._track_variable(self.prior_logits)
        else:
            self.prior = tf.constant(tf.ones([self.num_clusters]) * (1 / self.num_clusters))

        # Loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.recon_loss_tracker = keras.metrics.Mean(name='recon_loss')
        self.negloglik_cont_tracker = keras.metrics.Mean(name='negloglik_cont')
        self.negloglik_bin_tracker = keras.metrics.Mean(name='negloglik_bin')
        self.gmm_loss_tracker = keras.metrics.Mean(name='gmm_loss')

        if self.classify:
            self.classif_loss_tracker = keras.metrics.Mean(name='classif_loss')

    @property
    def metrics(self):
        loss_tracking = [
            self.negloglik_cont_tracker,
            self.negloglik_bin_tracker,
            self.gmm_loss_tracker,
            self.recon_loss_tracker,
            self.total_loss_tracker
            ]
        if self.classify:
            loss_tracking.append(self.classif_loss_tracker)

        return loss_tracking

    def call(self, data):
        '''Predict/call method.'''
        z_mean, z_logvar, z = self.encoder(data)

        x_mean_pred, x_logvar_pred, theta_pred = self.decoder(z_mean)
        reconstructed = tf.concat([x_mean_pred, theta_pred], axis=1) # Reconstruction = Mean (or prob.) of estimated distribution
        c_samples = self.get_clusters(z_mean) # Hard cluster assignments
        if self.classify:
            if self.s_to_classifier:
                c_onehot = tf.one_hot(c_samples, depth=self.num_clusters)
                # classif_inp = tf.concat([z, tf.one_hot(c_samples, depth=self.num_classes)], axis=-1)
                classif_inp = (z_mean, c_onehot)
            else:
                classif_inp = z_mean
            if self.classif_dependent:
                pred_label1, pred_label2, pred_label3 = self.classifier(classif_inp)
                if self.num_output_head == 1:
                    return reconstructed, z_mean, c_samples, pred_label1
                if self.num_output_head == 2:
                    return reconstructed, z_mean, c_samples, pred_label1, pred_label2
                else:
                    return reconstructed, z_mean, c_samples, pred_label1, pred_label2, pred_label3
            else:
                pred_label1 = self.classifier1(classif_inp) 
                pred_label2 = self.classifier2(classif_inp)
                pred_label3 = self.classifier3(classif_inp)
                return reconstructed, z_mean, c_samples, pred_label1, pred_label2, pred_label3
        return reconstructed, z_mean, c_samples
    
    def compute_loss(self, inputs):
        '''Computes negative augmented ELBO, that is minimized when optimizing.'''
        if self.classify:
            if self.num_output_head == 1:
                data, missing_mask, true_label1 = inputs
            elif self.num_output_head == 2:
                data, missing_mask, (true_label1, true_label2)= inputs
            elif self.num_output_head == 3:
                # data, missing_mask, [true_label1, true_label2, true_label3] = inputs[0]
                data, missing_mask, (true_label1, true_label2, true_label3) = inputs
        else:
            data, missing_mask = inputs

        z_mean, z_logvar, z = self.encoder(data)
        x_mean_pred, x_logvar_pred, theta_pred = self.decoder(z)

        # Separate continuous and categorical parts from input and reconstruction
        if self.cont_dim > 0:
            negloglik_cont = self.negloglik_gaussian(x_mean_pred, x_logvar_pred, data, missing_mask)
        else:
            negloglik_cont = 0

        if self.bin_dim > 0:
            negloglik_bin = self.negloglik_bernoulli(theta_pred, data, missing_mask)
        else:
            negloglik_bin = 0

        # Combine reconstruction losses
        recon_loss = tf.reduce_mean(negloglik_cont + negloglik_bin) # Term 1 (negative version)
        # GMM losses (sum of loss terms 3-6)
        gmm_loss = self.gmm_loss(z, z_logvar) # Terms 3-6

        tf.debugging.check_numerics(negloglik_cont, "Cont recon loss has NaN or Inf!")
        tf.debugging.check_numerics(negloglik_bin, "Bin recon loss has NaN or Inf!")
        tf.debugging.check_numerics(recon_loss, "Reconstruction loss has NaN or Inf!")
        tf.debugging.check_numerics(gmm_loss, "GMM loss has NaN or Inf!")

        if self.classify:
            if self.s_to_classifier:
                c_onehot = self.get_clusters(z)
                c_samples = tf.one_hot(c_onehot, depth=self.num_clusters)
                classif_inp = (z, c_samples)
            else:
                classif_inp = z
            if self.classif_dependent:
                pred_label1, pred_label2, pred_label3 = self.classifier(classif_inp)

                classif_loss = self.classification_loss(true_label1, pred_label1) # Term 2 (negative version)
                if self.num_output_head > 1:
                    classif_loss += self.classification_loss(true_label2, pred_label2) # Term 2 (negative version)
                if self.num_output_head > 2:
                    classif_loss += self.classification_loss(true_label3, pred_label3) 
            else:
                pred_label1 = self.classifier1(classif_inp) 
                pred_label2 = self.classifier2(classif_inp)
                pred_label3 = self.classifier3(classif_inp)
                classif_loss = self.classification_loss(true_label1, pred_label1)
                classif_loss += self.classification_loss(true_label2, pred_label2)
                classif_loss += self.classification_loss(true_label3, pred_label3)
            
            tf.debugging.check_numerics(classif_loss, "Classif loss has NaN or Inf!")
            
            # total_loss = tf.reduce_mean(recon_loss + classif_loss + gmm_loss) # This is the Negative ELBO + Classification loss
            total_loss = recon_loss + self.gamma * classif_loss + gmm_loss # This is the Negative ELBO + Classification loss
            tf.debugging.check_numerics(total_loss, "Total loss has NaN or Inf!")
            return total_loss, recon_loss, gmm_loss, classif_loss, negloglik_cont, negloglik_bin
        
        # total_loss = tf.reduce_mean(recon_loss + gmm_loss) # -ELBO
        total_loss = recon_loss + gmm_loss # -ELBO
        return total_loss, recon_loss, gmm_loss, negloglik_cont, negloglik_bin

    def negloglik_gaussian(self, x_mean_pred, x_logvar_pred, data, missing_mask):
        '''Reconstruction loss of continuous variables.'''
        present_mask_cont = 1 - missing_mask[:, :self.cont_dim]
        negloglik = 0.5 * tf.square(data[:, :self.cont_dim] - x_mean_pred)/tf.exp(x_logvar_pred) + x_logvar_pred
        return tf.reduce_sum(negloglik * present_mask_cont, axis=1)

    def negloglik_bernoulli(self, theta_pred, data, missing_mask):
        '''Reconstruction loss of binary variables.'''
        tf.debugging.check_numerics(theta_pred, "theta pred has NaN or Inf!")
        tf.debugging.check_numerics(data[:, self.cont_dim:], "Binary data has NaN or Inf!")
        tf.debugging.assert_non_negative(data[:, self.cont_dim:], "Binary data has negative values!")
        tf.debugging.assert_less_equal(data[:, self.cont_dim:], tf.ones_like(data[:, self.cont_dim:]), "Binary data has values > 1!")
        tf.debugging.check_numerics(missing_mask, "missing_mask has NaN or Inf!")
        tf.debugging.assert_non_negative(theta_pred, "theta_pred has negative values!")
        tf.debugging.assert_less_equal(theta_pred, tf.ones_like(theta_pred), "theta_pred has values > 1!")
        present_mask_bin = 1 - missing_mask[:, self.cont_dim:]
        recon_bin = tf.clip_by_value(theta_pred, 1e-6, 1.0 - 1e-6) # adjusting clipping boundaries fixed NaN loss issue

        bin_crossentropy = -(data[:, self.cont_dim:] * tf.math.log(recon_bin) + 
                                (1 - data[:, self.cont_dim:]) * tf.math.log(1 - recon_bin))
        tf.debugging.check_numerics(present_mask_bin, "present mask has NaN or Inf!")
        tf.debugging.check_numerics(bin_crossentropy, "bin_crossentropy has NaN or Inf!")
        negloglik_bin = tf.reduce_sum(bin_crossentropy * present_mask_bin, axis=1)
        tf.debugging.check_numerics(negloglik_bin, "Bernoulli func: Bin recon loss has NaN or Inf!")
        return negloglik_bin

    def classification_loss(self, true_label, pred_label):
        '''Classfication loss.'''
        if self.final_activation == 'softmax': # Categorical (Binary treated as 2-class categorical)
            return keras.losses.CategoricalCrossentropy()(true_label, pred_label)
        elif self.final_activation == 'sigmoid': # Binary
            return keras.losses.BinaryCrossentropy()(true_label, pred_label)
    
    def gmm_loss(self, z, z_logvar):
        '''GMM losses & approximate posterior losses.'''
        # c_sigma = tf.math.exp(self.log_c_sigma)
        # Calculate log(p(z|c)) for each cluster using Gaussian log probability
        p_z_c = tf.stack([self.gaussian_log_prob(z, self.c_mu[i, :], self.log_c_sigma[i, :]) for i in range(self.num_clusters)], axis=-1)
        # p(c)
        if self.learn_prior:
            prior_logits = tf.math.abs(self.prior_logits)
            norm = tf.math.reduce_sum(prior_logits, keepdims=True)
            prior = prior_logits / (norm + 1e-60)
        else:
            prior = self.prior

        # p(c|z)
        p_c_z = tf.math.log(prior + 1e-60) + p_z_c
        p_c_z = tf.math.exp(tf.nn.log_softmax(p_c_z, axis=-1))
    
        ## Clustering loss
        loss_clustering = - tf.reduce_sum(tf.multiply(p_c_z, p_z_c), axis=-1) # Term 3 (negative version)
        ## Prior loss
        loss_prior = - tf.reduce_sum(tf.math.xlogy(p_c_z, 1e-60 + prior), axis=-1) # Term 4 (negative version)
        ## Variational posterior loss components
        loss_variational_1 = - 0.5 * tf.reduce_sum(z_logvar + 1 + tf.math.log(2.0 * tf.constant(np.pi)), axis=-1) # Term 5 (negative version)
        loss_variational_2 = tf.reduce_sum(tf.math.xlogy(p_c_z, 1e-60 + p_c_z), axis=-1) # Term 6 (negative version)

        tf.debugging.check_numerics(loss_clustering, "Clustering loss has NaN or Inf!")
        tf.debugging.check_numerics(loss_prior, "Prior loss has NaN or Inf!")
        tf.debugging.check_numerics(loss_variational_1, "Variational 1 loss has NaN or Inf!")
        tf.debugging.check_numerics(loss_variational_2, "Variational 2 loss has NaN or Inf!")

        return tf.reduce_mean(loss_clustering + loss_prior + loss_variational_1 + loss_variational_2)

    def gaussian_log_prob(self, x, mean, log_sigma):
        """Calculates the log probability of x under a Gaussian distribution with given mean and log sigma."""
        return -0.5 * tf.reduce_sum(((x - mean) ** 2) / tf.exp(log_sigma) + log_sigma + tf.math.log(2.0 * tf.constant(np.pi)), axis=-1)

    def get_clusters(self, z, probs=False):
        '''Get hard cluster assignments (by default).'''
        # c_sigma = tf.math.exp(self.log_c_sigma)
        # Calculate log(p(z|c)) for each cluster using Gaussian log probability
        p_z_c = tf.stack([self.gaussian_log_prob(z, self.c_mu[i, :], self.log_c_sigma[i, :]) for i in range(self.num_clusters)], axis=-1)
        # p(c)
        if self.learn_prior:
            prior_logits = tf.math.abs(self.prior_logits)
            norm = tf.math.reduce_sum(prior_logits, keepdims=True)
            prior = prior_logits / (norm + 1e-60)
        else:
            prior = self.prior

        # p(c|z)
        p_c_z = tf.math.log(prior + 1e-60) + p_z_c
        p_c_z = tf.math.exp(tf.nn.log_softmax(p_c_z, axis=-1))
        # return tf.cast(tf.math.argmax(p_c_z, axis=-1), tf.float32)
        if probs:
            return p_c_z
        else:
            return tf.math.argmax(p_c_z, axis=-1)
    
    def train_step(self, inputs):
        '''Custom train step.'''
        with tf.GradientTape() as tape:
            if self.classify:
                total_loss, recon_loss, gmm_loss, classif_loss, negloglik_cont, negloglik_bin = self.compute_loss(inputs)
            else:
                total_loss, recon_loss, gmm_loss, negloglik_cont, negloglik_bin = self.compute_loss(inputs)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.negloglik_cont_tracker.update_state(negloglik_cont)
        self.negloglik_bin_tracker.update_state(negloglik_bin)
        self.gmm_loss_tracker.update_state(gmm_loss)

        losses = {
            "negloglik_cont": self.negloglik_cont_tracker.result(),
            "negloglik_bin": self.negloglik_bin_tracker.result(),
            "gmm_loss": self.gmm_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result()
            }

        if self.classify:
            self.classif_loss_tracker.update_state(classif_loss)
            losses['classif_loss'] = self.classif_loss_tracker.result()

        return losses
    
    def test_step(self, inputs):
        '''Custom test step, used when validating the model e.g. for learning rate decay.'''
        if self.classify:
            total_loss, recon_loss, gmm_loss, classif_loss, negloglik_cont, negloglik_bin = self.compute_loss(inputs)
        else:
            total_loss, recon_loss, gmm_loss, negloglik_cont, negloglik_bin = self.compute_loss(inputs)

        tf.debugging.check_numerics(total_loss, "Validation total loss has NaN or Inf!")

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(tf.reduce_mean(recon_loss))
        self.negloglik_cont_tracker.update_state(tf.reduce_mean(negloglik_cont))
        self.negloglik_bin_tracker.update_state(tf.reduce_mean(negloglik_bin))
        self.gmm_loss_tracker.update_state(tf.reduce_mean(gmm_loss))

        losses = {
            "negloglik_cont": self.negloglik_cont_tracker.result(),
            "negloglik_bin": self.negloglik_bin_tracker.result(),
            "gmm_loss": self.gmm_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result()
            }

        if self.classify:
            self.classif_loss_tracker.update_state(tf.reduce_mean(classif_loss))
            losses['classif_loss'] = self.classif_loss_tracker.result()

        return losses


    def generate(self, n_samples = 50, cluster=None):
        '''Generate new samples (not implemented).'''
        print('Generative property not implemented.')
        pass
