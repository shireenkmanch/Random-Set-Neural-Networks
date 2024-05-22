import tensorflow as tf
import keras
import keras.backend as K
from rsnn_functions.belief_mass_betp import mass_coeff
import numpy as np

new_classes = np.load('new_classes.npy', allow_pickle=True)
# mass_coeff_matrix = mass_coeff(new_classes)
# mass_coeff_matrix = tf.cast(mass_coeff_matrix, tf.float32)

ALPHA = 0.001
BETA = 0.001

@keras.saving.register_keras_serializable(package="my_package", name="BinaryCrossEntropy")

def BinaryCrossEntropy(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)
  y_true = K.clip(y_true, K.epsilon(), 1)
  y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
  term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  
  term_1 = y_true * K.log(y_pred + K.epsilon())
  bce_loss = -K.mean(term_0 + term_1, axis=0)
  
  mass_coeff_matrix = mass_coeff(new_classes)
  mass_coeff_matrix = tf.cast(mass_coeff_matrix, tf.float32)
  mass = tf.matmul(y_pred, mass_coeff_matrix)

  # alpha = tf.cast(tf.where(mass >= 0, tf.ones_like(mass), tf.zeros_like(mass)), dtype=tf.float32)
  
  mass_reg = K.mean(tf.nn.relu(-mass))

  mass_sum = tf.nn.relu(K.mean(K.sum(mass, axis=-1)) - 1)
  
  #add alpha to bce term 1 or 2
  # alpha_reg = -K.mean((1 - alpha) * K.log(1 - y_true + K.epsilon()) + alpha * K.log(y_true + K.epsilon()), axis = 0)
  

  total_loss = bce_loss + ALPHA * mass_reg + BETA * mass_sum
  # tf.print(K.mean(bce_loss), K.sum(mass_reg), K.mean(total_loss))

  return total_loss