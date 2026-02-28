# IMPORTS
import tensorflow as tf
import tf_keras as keras
import tensorflow_probability as tfp

from tf_keras.models import Sequential, Model
from tf_keras.layers import Input, Dense, BatchNormalization, Dropout
from tf_keras.optimizers import SGD, Adam
import tensorflow_probability as tfp
from tf_keras import metrics
from tf_keras.callbacks import Callback
from sklearn.metrics import f1_score
from ray import train
from sklearn.preprocessing import RobustScaler
import numpy as np


# Set seed from random number generator, for better comparisons
from numpy.random import seed
seed(123)

import matplotlib.pyplot as plt



# Source - https://stackoverflow.com/a/45305384
# Posted by Vadim, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-25, License - CC BY-SA 4.0


## Since keras does not compute the Macro F1 score correctly, a custom implementation is requited
## Implementation taken from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

# def f1(y_true, y_pred):
#     def recall(y_true, y_pred):
#         """Recall metric.

#         Only computes a batch-wise average of recall.

#         Computes the recall, a metric for multi-label classification of
#         how many relevant items are selected.
#         """
#         true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
#         possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + backend.epsilon())
#         return recall

#     def precision(y_true, y_pred):
#         """Precision metric.

#         Only computes a batch-wise average of precision.

#         Computes the precision, a metric for multi-label classification of
#         how many selected items are relevant.
#         """
#         true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + backend.epsilon())
#         return precision
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+backend.epsilon()))


class MacroF1Callback(Callback):
    def __init__(self, val_data, test_data=None):
        self.X_val, self.Y_val = val_data
        self.test_data = test_data
    
    def on_epoch_end(self, epoch, logs=None):
        preds = (self.model.predict(self.X_val, verbose=0) > 0.5).astype(int)
        score = f1_score(self.Y_val, preds, average='macro')
        logs['val_macro_f1'] = score
        print(f" — val_macro_f1: {score:.4f}")
    
    def on_test_end(self, logs=None):
        if self.test_data is not None:
            X_test, Y_test = self.test_data
            preds = (self.model.predict(X_test, verbose=0) > 0.5).astype(int)
            score = f1_score(Y_test, preds, average='macro')
            logs['test_macro_f1'] = score
            print(f" — test_macro_f1: {score:.4f}")


def normalize(Xtrain, Xval, Xtest, method = 'z-score'):
    if method == 'z-score':
        # Calculate normalization parameters
        mean_Xtrain = np.mean(Xtrain, axis=0)
        std_Xtrain = np.std(Xtrain, axis=0)

        # Remove mean of each covariate (column)
        Xtrain = Xtrain - mean_Xtrain
        Xtest = Xtest - mean_Xtrain
        Xval = Xval - mean_Xtrain

        # Divide each covariate (column) by its standard deviation
        Xtrain = Xtrain / std_Xtrain
        Xtest = Xtest / std_Xtrain
        Xval = Xval / std_Xtrain
    elif method == 'RobustScaler':
        # Calculate normalization parameters
        normalizer = RobustScaler().fit(Xtrain)

        # Apply normalization parameters
        XtrainRobust = normalizer.transform(Xtrain)
        XtestRobust = normalizer.transform(Xtest)
        XvalRobust = normalizer.transform(Xval)
    
    else:
        raise("The specified method is not supported. Use 'z-score' or 'RobustScaler'")

    return Xtrain, Xval, Xtest


def remove_outliers(X, Y, method = 'z-score', threshold = 5):
    if method == 'z-score':
        # calculate mean and standard deviation of the dataset
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = np.finfo(np.float64).eps

        # calculate the Z-Score
        z_scores = np.abs((X - mean) / std)

        # identify the outliers
        outliers = np.unique(np.where(z_scores > threshold)[0])
    elif method == 'IQR':
        Q1 = np.percentile(X, 25, method='midpoint', axis=0)
        Q3 = np.percentile(X, 75, method='midpoint', axis=0)
        IQR = Q3 - Q1
        upper = Q3+threshold*IQR
        lower = Q1-threshold*IQR

        xi_outliers = np.where((X[:,0] > upper[0]) | (X[:,0] < lower[0]))
        outliers = xi_outliers[0]

        outliers = np.unique(np.where((X > upper) | (X < lower))[0])  

    elif method == 'MAD':
        # calculate the median of the dataset
        median = np.median(X, axis = 0)

        # calculate the absolute deviations from the median
        abs_dev = np.abs(X - median)

        # calculate the median absolute deviation
        mad = np.median(abs_dev, axis = 0)
        mad[mad == 0] = np.finfo(np.float64).eps  #when mad is zero

        # identify the outliers
        outliers = np.unique(np.where(abs_dev / mad > threshold)[0])

    elif method == 'modified z-score':
        # calculate the median and the median absolute deviation
        median = np.median(X, axis=0)
        abs_dev = np.abs(X - median)
        mad = np.median(abs_dev, axis = 0)
        mad[mad == 0] = np.finfo(np.float64).eps  #when mad is zero

        # calculate the modified Z-Score for each value
        modified_zscores = 0.6745 * abs_dev / mad

        # identify the outliers
        outliers = np.unique(np.where(modified_zscores > threshold)[0])
    
    Xclean = np.delete(X, outliers, axis=0)
    Yclean = np.delete(Y, outliers)
    print(outliers.shape[0], 'outliers removed')

    return Xclean, Yclean





# =======================================
# DNN related function
# =======================================
# DEEP LEARNING MODEL BUILD FUNCTION
def build_DNN(input_shape, n_hidden_layers, n_hidden_units, loss, act_fun='sigmoid', optimizer:str='sgd', learning_rate=0.01, 
            use_bn=False, use_dropout=False, use_custom_dropout=False, print_summary=False, use_variational_layer=False, kl_weight = 1/53.5):
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
    if optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    # else:
    #     opt = None
    
    f1 = keras.metrics.F1Score(average='macro', threshold=0.5, name="f1_score", dtype=None)


    # ============================================

    # Setup a sequential model
    model = Sequential()

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    # Add layers to the model, using the input parameters of the build_DNN function
    
    # Add first (Input) layer, requires input shape
    if use_variational_layer:
        model.add(tfp.layers.DenseVariational(n_hidden_units, make_prior_fn = prior, make_posterior_fn = posterior, activation = act_fun, kl_weight = kl_weight, input_dim = input_shape ))
    else:
        model.add(Dense(n_hidden_units, activation= act_fun, input_dim = input_shape))
    if use_dropout:
        model.add(Dropout(0.5))
    if use_custom_dropout:
        model.add(myDropout(0.5))
    
    if use_bn:
        model.add(BatchNormalization())
    # Add remaining layers. These to not require the input shape since it will be infered during model compile
    for _ in range(n_hidden_layers-1):
        if use_variational_layer:
            model.add(tfp.layers.DenseVariational(n_hidden_units, make_prior_fn = prior, make_posterior_fn = posterior, activation = act_fun, kl_weight = kl_weight))
        else:
            model.add(Dense(n_hidden_units, activation=act_fun))
        if use_dropout:
            model.add(Dropout(0.5))
        
        if use_custom_dropout:
            model.add(myDropout(0.5))
             
        if use_bn:
            model.add(BatchNormalization())

         
    # Add final layer
    if use_variational_layer:
        model.add(tfp.layers.DenseVariational(1, make_prior_fn = prior, make_posterior_fn = posterior, activation = 'sigmoid', kl_weight = kl_weight))
    else:
        model.add(Dense(1, activation='sigmoid'))

    
    
    # Compile model
    model.compile(loss = loss, optimizer= opt,  metrics = ['accuracy', f1])

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
            train.report(dict(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss")))
    
    # --------------------------------------------  
    # === Your code here =========================
    # --------------------------------------------
    # Unpack the data tuple
    X_train, y_train, X_val, y_val = training_config['data']

    # Build the model using the variables stored into the config dictionary.
    # Hint: you provide the config dictionary to the build_DNN function as a keyword argument using the ** operator.
    model = build_DNN(**config)
        
    # Train the model (no need to save the history, as the callback will log the results).
    # Remember to add the TuneReporterCallback() to the list of callbacks.
    model.fit(x = X_train, y = y_train, batch_size= training_config['batch_size'], epochs= training_config['epochs'], validation_data= (X_val, y_val), callbacks=[TuneReporterCallback()])

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
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
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
def plot_results(history, metric = 'accuracy'):
    """
    Plots the training and validation loss and accuracy from a Keras history object.
    Parameters:
    history (keras.callbacks.History): A History object returned by the fit method of a Keras model. 
                                       It contains the training and validation loss and accuracy for each epoch.
    Returns:
    None
    """
    
    val_loss = history.history['val_loss']
    acc = history.history[metric]
    loss = history.history['loss']
    val_acc = history.history[f'val_{metric}']
    
    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training','Validation'])

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training','Validation'])

    plt.show()
