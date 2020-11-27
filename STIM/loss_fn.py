import tensorflow as tf
import scipy as sp
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layer
import tensorflow.keras.backend as K
import time, random, threading
import numpy as np
import gc

EPSILON = 1e-20

'''
Implementação de funções de perda que otimizam o ranking ao invés
de otimizar o erro direto. Ver o seguinte artigo para mais detalhes:
- "SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS"
'''

def bpr_loss(x_pos, x_neg):
	return -tf.reduce_mean(tf.log(tf.sigmoid(x_pos - x_neg) + EPSILON))

#------------------------------------------------------------------------------
def top1_loss(x_pos, x_neg):
	return tf.reduce_mean(tf.sigmoid(x_neg - x_pos) + tf.sigmoid(x_neg))

#------------------------------------------------------------------------------
def cross_entropy_loss(y_true, y_pred):
	return -tf.reduce_sum(y_true * tf.log(y_pred))

#------------------------------------------------------------------------------
def bce(y_true, y_pred):
	return tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true)

#------------------------------------------------------------------------------
def bce_smooth(y_true, y_pred):
	return tf.contrib.losses.sigmoid_cross_entropy(logits=y_pred, multi_class_labels=y_true, label_smoothing=constants.SMOOTH_FACTOR)

#------------------------------------------------------------------------------
def binary_focal_loss_sigmoid(y_true, y_pred):
	"""
	Binary form of focal loss.
		FL(p_t) = -ALPHA * (1 - p_t)**GAMMA * log(p_t)
		where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
	References:
		https://arxiv.org/pdf/1708.02002.pdf
	Usage:
		model.compile(loss=[binary_focal_loss(ALPHA=.25, GAMMA=2)], metrics=["accuracy"], optimizer=adam)
		
	:param y_true: A tensor of the same shape as `y_pred`
	:param y_pred:  A tensor resulting from a sigmoid
	:return: Output tensor.
	"""
	pred = tf.sigmoid(y_pred)
	pt_1 = tf.where(tf.equal(y_true, 1), pred, tf.ones_like(pred))
	pt_0 = tf.where(tf.equal(y_true, 0), pred, tf.zeros_like(pred))

	epsilon = 1e-18
	# clip to prevent NaN's and Inf's
	pt_1 = tf.clip_by_value(pt_1, epsilon, 1.0)
	pt_0 = tf.clip_by_value(pt_0, epsilon, 1.0)

	return -K.sum(0.25 * K.pow(1. - pt_1, 2) * K.log(pt_1)) \
			-K.sum((1 - 0.25) * K.pow(pt_0, 2) * K.log(1. - pt_0))

#------------------------------------------------------------------------------
def binary_focal_loss_softmax(y_true, y_pred):
	pred = tf.nn.softmax(y_pred)
	pt_1 = tf.where(tf.equal(y_true, 1), pred, tf.ones_like(pred))
	pt_0 = tf.where(tf.equal(y_true, 0), pred, tf.zeros_like(pred))

	epsilon = 1e-18
	# clip to prevent NaN's and Inf's
	pt_1 = tf.clip_by_value(pt_1, epsilon, 1.0)
	pt_0 = tf.clip_by_value(pt_0, epsilon, 1.0)

	return -K.sum(0.25 * K.pow(1. - pt_1, 2) * K.log(pt_1)) \
			-K.sum((1 - 0.25) * K.pow(pt_0, 2) * K.log(1. - pt_0))
