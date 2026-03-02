# IMPORTS
import tensorflow as tf
import tf_keras as keras
import tensorflow_probability as tfp

from tf_keras.models import Sequential, Model
from tf_keras.layers import Input, Dense, BatchNormalization, Dropout
from tf_keras.optimizers import SGD, Adam
import tensorflow_probability as tfp

from ray import train
from ray import tune

import math
from tf_keras.callbacks import LearningRateScheduler

import numpy as np
from sklearn.utils import class_weight

# Set seed from random number generator, for better comparisons
from numpy.random import seed
seed(123)

import matplotlib.pyplot as plt

# =======================================
# DNN related function
# =======================================
# DEEP LEARNING MODEL BUILD FUNCTION
def build_DNN(input_shape, n_hidden_layers, n_hidden_units, loss, act_fun='sigmoid', optimizer:str='sgd', learning_rate=0.01, 
            use_bn=False, use_dropout=False, use_custom_dropout=False, print_summary=False, use_variational_layer=False, global_clipnorm=10**10, kl_weight = 100):
    """
    Builds a Deep Neural Network (DNN) model based on the provided parameters.
    
    Parameters:
    input_shape (tuple): Shape of the input data (excluding batch size).
    n_hidden_layers (int): Number of hidden layers in the model.
    n_hidden_units (int): Number of nodes in each hidden layer (here all hidden layers have the same shape).
    loss (keras.losses): Loss function to use in the model.
    act_fun (str, optional): Activation function to use in each layer. Default is 'sigmoid'.
    optimizer (str, optional): Optimizer to use in the model. Default is SGD.
    learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
    use_bn (bool, optional): Whether to use Batch Normalization after each layer. Default is False.
    use_dropout (bool, optional): Whether to use Dropout after each layer. Default is False.
    use_custom_dropout (bool, optional): Whether to use a custom Dropout implementation. Default is False.
    
    Returns:
    model (Sequential): Compiled Keras Sequential model.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------      
    # Setup optimizer, depending on input parameter string  


    if optimizer == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate = learning_rate, global_clipnorm=global_clipnorm)
    elif optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate = learning_rate, global_clipnorm=global_clipnorm)
    else:
        raise ValueError("Suppported optimizers: sgd, adam")


    # ============================================

    # Setup a sequential model
    model = Sequential()

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    # Add layers to the model, using the input parameters of the build_DNN function
    

    if use_variational_layer:
        model.add(tfp.layers.DenseVariational(n_hidden_units, activation = act_fun, input_shape = input_shape, kl_weight=kl_weight,
                                              make_prior_fn = prior, make_posterior_fn = posterior))
    else:
        model.add(Dense(n_hidden_units, activation = act_fun, input_shape = input_shape))
    
    if type(use_dropout) == float or type(use_dropout) == int:
        model.add(Dropout(use_dropout))
    if type(use_custom_dropout) == float or type(use_custom_dropout) == int:
        model.add(myDropout(use_custom_dropout))
    if use_bn: 
        model.add(BatchNormalization())
        

        
        # Add remaining layers. These to not require the input shape since it will be infered during model compile
    for _ in range(n_hidden_layers - 1):
        if use_variational_layer:
            model.add(tfp.layers.DenseVariational(n_hidden_units, activation = act_fun, kl_weight=kl_weight,
                                              make_prior_fn = prior, make_posterior_fn = posterior))
        else:
            model.add(Dense(n_hidden_units, activation = act_fun))
        if type(use_dropout) == float or type(use_dropout) == int:
            model.add(Dropout(use_dropout))
        if type(use_custom_dropout) == float or type(use_custom_dropout) == int:
            model.add(myDropout(use_custom_dropout))
        if use_bn: 
            model.add(BatchNormalization())



    
         
    # Add final layer
    if use_variational_layer:
        model.add(tfp.layers.DenseVariational(1, activation = "sigmoid", kl_weight=kl_weight,
                                              make_prior_fn = prior, make_posterior_fn = posterior))
    else:
        model.add(Dense(1, activation = "sigmoid"))

    
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    # ============================================
    # Print model summary if requested
    if print_summary:
        model.summary() 
    
    return model

def train_DNN(config, training_config):
    '''
    Train a DNN model based on the provided configuration and data. 
    This is use in the automatic hyperparameter search and follows the format that Ray Tune expects.

    Parameters:
    config (dict): Dictionary with the configuration parameters for the model. This includes the parameters needed to build the model and can be 
                    manually set or generated by Ray Tune.
                    For convenience, the config dictionary also contains the training parameters, such as the number of epochs and batch size.
    training_config (dict): Dictionary with the training parameters, such as the number of epochs and batch size, and the data to use for training and validation (Xtrain, Ytrain, Xval, Yval).
    '''

    # A dedicated callback function is needed to allow Ray Tune to track the training process
    # This callback will be used to log the loss and accuracy of the model during training
    class TuneReporterCallback(keras.callbacks.Callback):
        """Tune Callback for Keras.
        
        The callback is invoked every epoch.
        """
        def __init__(self, logs={}):
            self.iteration = 0
            super(TuneReporterCallback, self).__init__()
    
        def on_epoch_end(self, batch, logs={}):
            self.iteration += 1
            tune.report(dict(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss")))
    
    # --------------------------------------------  
    # === Your code here =========================
    # --------------------------------------------
    # Unpack the data tuple
    X_train, y_train, X_val, y_val = training_config["data"]

    # Build the model using the variables stored into the config dictionary.
    # Hint: you provide the config dictionary to the build_DNN function as a keyword argument using the ** operator.



    batch_size = training_config["batch_size"]
    epochs = training_config["epochs"]
    learning_rate = config["learning_rate"]
    loss = tf.keras.losses.BinaryFocalCrossentropy(gamma = config["gamma"], label_smoothing=0.02)
    model = build_DNN(config["input_shape"], config["n_hidden_layers"], config["n_hidden_units"], loss, act_fun=config["act_fun"], optimizer=config["optimizer"], learning_rate=config["learning_rate"], print_summary=False, 
                       use_bn=config["use_bn"], use_dropout=config["use_dropout"], use_custom_dropout=config["use_custom_dropout"])

    #model=build_DNN(hyperparameter_space**) this doesnt work well with gamma and max_lr.
        
    # Train the model (no need to save the history, as the callback will log the results).
    # Remember to add the TuneReporterCallback() to the list of callbacks.

    max_lr = config["max_lr"]
    warmup_duration = round(epochs*0.2)
    # cosine annealing, one cycle
    #https://wiki.cloudfactory.com/docs/mp-wiki/scheduler/cosineannealinglr
    def scheduler(epoch, lr):
        if epoch < warmup_duration:
            return learning_rate + epoch * (max_lr-learning_rate) / warmup_duration
        else:
            decay_duration = epochs - warmup_duration
            epochs_past_warmup = epoch - warmup_duration
            min_lr = learning_rate * 0.1 

            progress = epochs_past_warmup / decay_duration
            cosine_multiplier = 0.5 * (1 + math.cos(math.pi * progress))
            
            return min_lr + (max_lr - min_lr) * cosine_multiplier    
        
    callback = LearningRateScheduler(scheduler)


    value1, value2 = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y= y_train)

    class_weights = {0: value1,
                1: value2}

    model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = epochs, batch_size = batch_size, class_weight = class_weights, 
                                callbacks=[callback, TuneReporterCallback()])


    
    # --------------------------------------------


# CUSTOM DROPOUT IMPLEMENTATION
# Code from https://github.com/keras-team/tf-keras/issues/81
class myDropout(keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)  # Override training=True


# CUSTOM PRIOR AND POSTERIOR FUNCTIONS FOR THE VARIATIONAL LAYER
#  Code from https://keras.io/examples/keras_recipes/bayesian_neural_networks/
# The prior is defined as a normal distribution with zero mean and unit variance.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Mixture(
                    cat = tfp.distributions.Categorical(probs=[0.16, 1 - 0.16]),
                    components = [
                        tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n)),
                        tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n))
                    ]
                )
                )
        ]
    )
    return prior_model


# multivariate Gaussian distribution parametrized by a learnable parameters.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


# =======================================
# PLOTTING FUNCTIONS
# =======================================

# TRAINING CURVES PLOT FUNCTION
def plot_results(history, lrbool = False):
    """
    Plots the training and validation loss and accuracy from a Keras history object.
    Parameters:
    history (keras.callbacks.History): A History object returned by the fit method of a Keras model. 
                                       It contains the training and validation loss and accuracy for each epoch.
    Returns:
    None
    """
    
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    if lrbool:
        lr = history.history['lr']

        plt.figure(figsize=(10,4))
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.plot(lr)
    
    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training','Validation'])

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training','Validation'])

    plt.show()
