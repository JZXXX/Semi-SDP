#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2017 Timothy Dozat
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import numpy as np
import tensorflow as tf

#***************************************************************
def get_sizes(t):
  """"""
  
  shape = []
  for i in six.moves.range(len(t.get_shape().as_list()[:-1])):
    shape.append(tf.shape(t)[i])
  shape.append(t.get_shape().as_list()[-1])
  return shape
  
#===============================================================
def reshape(t, shape):
  """"""
  
  if isinstance(shape, (tuple, list)):
    shape = tf.stack(shape)
  return tf.reshape(t, shape)

#===============================================================
def orthogonal_loss(x):
  """"""
  
  output_size = x.get_shape().as_list()[-1]
  x = tf.reshape(x, [-1, output_size])
  input_size = tf.shape(x)[0]
  I = tf.eye(output_size)
  xTx = tf.matmul(x, x, transpose_a=True)
  off_diag_xTx = xTx * (1-I)
  loss = tf.nn.l2_loss(off_diag_xTx)
  return loss

#===============================================================
def dropout(inputs, keep_prob, noise_shape=None):
  """"""
  
  if isinstance(noise_shape, (tuple, list)):
    noise_shape = tf.stack(noise_shape)
  return tf.nn.dropout(inputs, keep_prob=keep_prob, noise_shape=noise_shape)

#===============================================================
def unscaled_dropout(inputs, keep_prob, noise_shape=None):
  """"""
  
  if isinstance(noise_shape, (tuple, list)):
    noise_shape = tf.stack(noise_shape)
  return tf.nn.dropout(inputs, keep_prob=keep_prob, noise_shape=noise_shape)*keep_prob

#===============================================================
def drop_mask(shape, keep_prob):
  """"""
  
  if isinstance(shape, (tuple, list)):
    shape = tf.stack(shape)
  ones = tf.ones(shape)
  return dropout(ones, keep_prob)

#===============================================================
def binary_mask(shape, keep_prob):
  """"""
  
  if isinstance(shape, (tuple, list)):
    shape = tf.stack(shape)
  ones = tf.ones(shape)
  return unscaled_dropout(ones, keep_prob)

#===============================================================
def greater_equal(input1, input2, dtype=None):
  
  ones = tf.ones_like(input1, dtype=dtype)
  zeros = tf.zeros_like(input1, dtype=dtype)
  return tf.where(tf.greater_equal(input1, input2), ones, zeros)

#===============================================================
def greater(input1, input2, dtype=None):
  
  ones = tf.ones_like(input1, dtype=dtype)
  zeros = tf.zeros_like(input1, dtype=dtype)
  # Return the elements, either from `x` or `y`, depending on the `condition`
  return tf.where(tf.greater(input1, input2), ones, zeros)

#===============================================================
def less_equal(input1, input2, dtype=None):
  
  ones = tf.ones_like(input1, dtype=dtype)
  zeros = tf.zeros_like(input1, dtype=dtype)
  return tf.where(tf.less_equal(input1, input2), ones, zeros)

#===============================================================
def less(input1, input2, dtype=None):
  
  ones = tf.ones_like(input1, dtype=dtype)
  zeros = tf.zeros_like(input1, dtype=dtype)
  return tf.where(tf.less(input1, input2), ones, zeros)

#===============================================================
def not_equal(input1, input2, dtype=None):
  
  ones = tf.ones_like(input1, dtype=dtype)
  zeros = tf.zeros_like(input1, dtype=dtype)
  return tf.where(tf.not_equal(input1, input2), ones, zeros)

#===============================================================
def equal(input1, input2, dtype=None):
  
  ones = tf.ones_like(input1, dtype=dtype)
  zeros = tf.zeros_like(input1, dtype=dtype)
  return tf.where(tf.equal(input1, input2), ones, zeros)

#===============================================================
def where(condition, x, y, dtype=tf.float32):
  
  ones = tf.ones_like(condition, dtype=dtype)
  return tf.where(condition, x*ones, y*ones)

#===============================================================
def ones(shape, dtype=tf.float32):
  
  if isinstance(shape, (tuple, list)):
    shape = tf.stack(shape)
  return tf.ones(shape, dtype=dtype)

#===============================================================
def zeros(shape, dtype=tf.float32):
  
  if isinstance(shape, (tuple, list)):
    shape = tf.stack(shape)
  return tf.zeros(shape, dtype=dtype)

#===============================================================
def tile(inputs, multiples):
  
  if isinstance(multiples, (tuple, list)):
    multiples = tf.stack(multiples)
  return tf.tile(inputs, multiples)

#----------------------------------------------------------------
def my_get_sizes(t):
  '''
  Function for retrieving the size of tensor inside a placeholder T.
  INPUT:
      T: The placeholder whose tensor size is going to be retrieved.

  RETURN:
      SHAPE: The shape of the tensor inside placeholder T.'''
  shape = []
  for i in range(len(t.get_shape().as_list())):
    shape.append(tf.shape(t)[i])
  return shape

#----------------------------------------------------------------
def shift_left_by_one(data, dim=1):
  '''
  Shift the tensor DATA by 1, on dimension DIM.
  INPUT:
       DATA: A placeholder which contains the data;
       DIM: dimension on which to shift.

   RETURN:
       SHIFTED: The tensor obtained by shifting DATA on DIM by 1.
   '''

  # get shape of the input tensor.
  data_shape = my_get_sizes(data)
  # keep those entries with index INDICES along DIM .
  indices = tf.range(1, data_shape[dim], delta=1)
  # The first vector along DIM as a reference for padding 0.
  ref = tf.gather(data, 0, axis=dim)
  # Shift the DATA tensor by keeping only [1:len(dim)].

  shifted = tf.gather(data, indices, axis=dim)
  # Pad 0 and expand its dimension to match with DATA.
  padding = tf.zeros_like(ref, dtype=ref.dtype)
  padding = tf.expand_dims(padding, dim)
  # Concat the shifted tensor and the padding vector together.
  shifted = tf.concat([shifted, padding], axis=dim)


  return shifted

#-----------------------------------------------------------------
def shift_right_by_one(data, dim=1):
  '''
  Right shift the tensor DATA by 1, on dimension DIM.
  INPUT:
       DATA: A placeholder which contains the data;
       DIM: dimension on which to shift.

   RETURN:
       SHIFTED: The tensor obtained by right-shifting DATA on DIM by 1.
   '''

  # get shape of the input tensor.
  data_shape = my_get_sizes(data)
  # keep those entries with index INDICES along DIM .
  indices = tf.range(0, data_shape[dim] - 1, delta=1)
  # The first vector along DIM as a reference for padding 0.
  ref = tf.gather(data, 0, axis=dim)
  # Shift the DATA tensor by keeping only [0:len(dim)-1].
  shifted = tf.gather(data, indices, axis=dim)
  # Pad 0 and expand its dimension to match with DATA.
  padding = tf.zeros_like(ref, dtype=ref.dtype)
  padding = tf.expand_dims(padding, dim)
  # Concat the shifted tensor and the padding vector together.
  shifted = tf.concat([padding, shifted], axis=dim)

  return shifted



# def shift_right_by_one(data, dim=1):
#   '''
#   Right shift the tensor DATA by 1, on dimension DIM.
#   INPUT:
#        DATA: A placeholder which contains the data;
#        DIM: dimension on which to shift.
#
#    RETURN:
#        SHIFTED: The tensor obtained by right-shifting DATA on DIM by 1.
#    '''
#
#   # get shape of the input tensor.
#   data_shape = my_get_sizes(data)
#   # keep those entries with index INDICES along DIM .
#   indices = tf.range(0, data_shape[dim] - 1, delta=1)
#   # The first vector along DIM as a reference for padding 0.
#   ref = tf.gather(data, data_shape[dim]-1, axis=dim)
#   ref = tf.expand_dims(ref, dim)
#   # Shift the DATA tensor by keeping only [0:len(dim)-1].
#   shifted = tf.gather(data, indices, axis=dim)
#   # Concat the shifted tensor and the padding vector together.
#   shifted = tf.concat([ref, shifted], axis=dim)
#
#   return shifted



#----------------------------------------------------------------
def select_idx(data, idx):
  '''
  :param data: placeholder data, i.e. shape[n,m,V]
  :param idx: placeholder idx, i.e. shape[n,m]
  :return: [n,m], from data[i,j,V] select value of [i,j] index from V
  '''
  shape = get_sizes(data)
  dim0 = 1
  for i in shape[:-1]:
    dim0 *= i
  data_reshape = reshape(data, [dim0, -1])
  idx_reshape = reshape(idx, [dim0])
  idx_ = tf.to_int64(tf.range(0, limit=dim0, delta=1, name='arange'))
  idx_ = tf.stack([idx_, idx_reshape], axis=-1)
  prob = tf.gather_nd(data_reshape, idx_)
  prob = reshape(prob, shape[:-1])
  return prob

#----------------------------------------------------------------
def expand_add(child,parent):
  '''
  :param child: n x m x dim
  :param parent: n x m x dim
  :return: n x m x m x dim
  '''
  shape = get_sizes(parent)
  c_tile = tf.tile(child, [1, shape[1], 1])
  c_tile = reshape(c_tile, [shape[0], shape[1], shape[1], shape[2]])
  p_tile = tf.tile(parent, [1, 1, shape[1]])
  p_tile = reshape(p_tile, [shape[0], shape[1], shape[1], shape[2]])
  return c_tile + p_tile

def expand_dims(child, parent):
  '''
    :param child: n x m x dim
    :param parent: n x m x dim
    :return: n x m x m x dim, n x m x m x dim
    '''
  shape = get_sizes(parent)
  c_tile = tf.tile(child, [1, shape[1], 1])
  c_tile = reshape(c_tile, [shape[0], shape[1], shape[1], shape[2]])
  p_tile = tf.tile(parent, [1, 1, shape[1]])
  p_tile = reshape(p_tile, [shape[0], shape[1], shape[1], shape[2]])
  return c_tile, p_tile

def expand_dim_noedge(noedge):

  shape = get_sizes(noedge)
  noedge = tf.expand_dims(noedge,1)
  noedge = tf.tile(noedge,[1,shape[-1],1])
  return noedge

def expand_dimp(parent):
  shape = get_sizes(parent)
  p_tile = tf.tile(parent, [1, 1, shape[1]])
  p_tile = reshape(p_tile, [shape[0], shape[1], shape[1], shape[2]])
  return p_tile

def expand_multiply(child,parent):
  '''
  :param child: n x m x dim
  :param parent: n x m x dim
  :return: n x m x m x dim
  '''
  shape = get_sizes(parent)
  c_tile = tf.tile(child, [1, shape[1], 1])
  c_tile = reshape(c_tile, [shape[0], shape[1], shape[1], shape[2]])
  p_tile = tf.tile(parent, [1, 1, shape[1]])
  p_tile = reshape(p_tile, [shape[0], shape[1], shape[1], shape[2]])
  return c_tile * p_tile

def omit_zeros(tensor):
  zero_vector = tf.zeros(shape=(1,1), dtype=tensor.dtype)
  bool_mask = tf.not_equal(tensor, zero_vector)
  omit_tensor = tf.boolean_mask(tensor,bool_mask)
  return  omit_tensor

def omit_inf(tensor):
  # inf_vector = tf.zeros(shape=(1,1), dtype=tensor.dtype)
  inf_vector = tf.math.log(0.)
  bool_mask = tf.not_equal(tensor, inf_vector)
  omit_tensor = tf.boolean_mask(tensor,bool_mask)
  return  omit_tensor

def zero_diag(tensor):
  shape = get_sizes(tensor)
  ones = tf.ones([shape[0],shape[1]])
  diag = tf.matrix_diag(ones)
  return  1-diag