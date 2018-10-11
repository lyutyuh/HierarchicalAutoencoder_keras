# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import sys
sys.path.append('..')
from keras.engine import Layer
import keras.backend as K
from keras.models import Sequential, Model
from keras import activations
from keras import initializers
from keras import optimizers
from keras.initializers import RandomUniform, TruncatedNormal
from keras.layers import Reshape, Embedding, Dot, Add, Multiply, \
    Input, Dense, Conv2D, CuDNNLSTM, Softmax, Lambda, Bidirectional, Concatenate
from keras.optimizers import Adam
from recurrentshop import LSTMCell
from models.model import BasicModel
from layers.Linear import Linear
import numpy as np
import tensorflow as tf
from collections import namedtuple



def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)
    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.
    Returns:
      a scalar
    """
    dec_lens = K.tf.reduce_sum(padding_mask, axis=1, name='dec_lens')  # shape batch_size. float32
    values_per_step = []
    
    n_tokens = values.get_shape()[1]
    
    for dec_step in range(n_tokens):
        v = values[:, dec_step]
        values_per_step.append(v * padding_mask[:, dec_step])

    sums = values_per_step[0]
    for val in values_per_step[1:]:
        sums += val
    values_per_ex = sums / (dec_lens+K.epsilon())  # shape (batch_size); normalized value for each batch member
    return K.tf.reduce_mean(values_per_ex, name='reduce_mean_in_mask_avg')  # overall average



def loss_wrapper(_targets, _mask):
    
    
    def _loss(y_true, y_pred):        
        nonlocal _targets, _mask     
        
        
        total_losses = K.sparse_categorical_crossentropy(_targets, y_pred)
        print('_targets',_targets.shape)
        print('y_pred',y_pred.shape)
        print('_mask', _mask.shape)
        print('total_losses', total_losses.shape)
        # n_samples  = K.cast(K.shape(total_losses)[1], dtype='float32')
        # sum_losses = K.sum(total_losses, axis=1)
        # sum_losses /= n_samples
        sum_losses = _mask_and_avg(total_losses, _mask)
        return sum_losses
        
    return _loss
