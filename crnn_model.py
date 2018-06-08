#!/usr/bin/env python
# -*- coding : utf-8 -*-

import tensorflow as tf 

def conv_layer(inputdata, kernel_size, out_channels, stride, padding, name):
    input_shape = inputdata.get_shape().as_list()
    in_channels = input_shape[3]
    kernel_size = kernel_size + [in_channels, out_channels]
    kernel_init = tf.contrib.layers.variance_scaling_initializer()
    bias_init = tf.constant_initializer()
    weight = tf.get_variable(name + '_weight', kernel_size, initializer=kernel_init)
    bias = tf.get_variable(name + '_bias', [out_channels], initializer=bias_init)
    strides = [1, stride[0], stride[1], 1]
    conv = tf.nn.conv2d(input=inputdata, filter=weight, strides=strides, padding=padding)
    conv = tf.nn.bias_add(conv, bias)
    return conv

def pooling_layer(inputdata, window_size, stride, padding='SAME'):
    window = [1, window_size[0], window_size[1], 1]
    strides = [1, stride[0], stride[1], 1]
    return tf.nn.max_pool(inputdata, ksize=window, strides=strides, padding=padding)

def relu_layer(inputdata):
    return tf.nn.relu(inputdata)

def batch_norm_layer(inputdata, axis=-1, training=False):
    return tf.layers.batch_normalization(inputdata, axis=axis, training=training)

def cnn_module(inputdata, training):
    with tf.variable_scope('cnn_module'):
        conv1 = conv_layer(inputdata, [3, 3], 64, [1, 1], 'SAME', 'conv1')
        relu1 = relu_layer(conv1)
        pool1 = pooling_layer(relu1, [2, 2], [2, 2], 'VALID')
        conv2 = conv_layer(pool1, [3, 3], 128, [1, 1], 'SAME', 'conv2')
        relu2 = relu_layer(conv2)
        pool2 = pooling_layer(relu2, [2, 2], [2, 2], 'VALID')
        conv3 = conv_layer(pool2, [3, 3], 256, [1, 1], 'SAME', 'conv3')
        relu3 = relu_layer(conv3)
        conv4 = conv_layer(relu3, [3, 3], 256, [1, 1], 'SAME', 'conv4')
        relu4 = relu_layer(conv4)
        pool4 = pooling_layer(relu4, [2, 2], [2, 2], 'VALID') # window = [2, 1]
        conv5 = conv_layer(pool4, [3, 3], 512, [1, 1], 'SAME', 'conv5')
        norm5 = batch_norm_layer(conv5, training=training)
        relu5 = relu_layer(norm5)
        conv6 = conv_layer(relu5, [3, 3], 512, [1, 1], 'SAME', 'conv6')
        norm6 = batch_norm_layer(conv6, training=training)
        relu6 = relu_layer(norm6)
        pool6 = pooling_layer(relu6, [2, 2], [2, 2], 'VALID') # window = [2, 1]
        conv7 = conv_layer(pool6, [2, 2], 512, [1, 1], 'VALID', 'conv7')
        relu7 = relu_layer(conv7)
        # map to sequence
        output = tf.squeeze(relu7, axis=1, name='sequence_feature')
        return output

def rnn_layer(inputdata, sequence_length, num_units, name):
    weight_init = tf.truncated_normal_initializer(stddev=0.01)
    fw_cell = tf.contrib.rnn.LSTMCell(num_units, initializer=weight_init)
    bw_cell = tf.contrib.rnn.LSTMCell(num_units, initializer=weight_init)
    rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, 
        inputdata, sequence_length, time_major=True, dtype=tf.float32, scope=name)
    # Concatenation allows a single output op because [A B]*[x;y] = Ax+By
    # [sequence_length, batch_size 2 * num_units]
    rnn_out_concat = tf.concat(rnn_out, axis=2, name='out_concat')
    return rnn_out_concat

def rnn_module(inputdata, num_units, num_classes):
    with tf.variable_scope('rnn_module'):
        # [batch, width(length), channels]
        [batch_size, sequence_length, channels] = inputdata.get_shape().as_list()
        sequence_feature = tf.transpose(inputdata, perm=[1, 0, 2], name='time_major') 
        sequence_length = [sequence_length] * batch_size    
        lstm1 = rnn_layer(sequence_feature, sequence_length, num_units, 'lstm1')        
        lstm2 = rnn_layer(lstm1, sequence_length, num_units, 'lstm2')
        rnn_logits = tf.layers.dense(lstm2, num_classes)
        return rnn_logits, sequence_length

def build_crnn(inputdata, training, num_units, num_classes):
    cnn_feature = cnn_module(inputdata, training)
    logits, sequence_length = rnn_module(cnn_feature, num_units, num_classes)
    return logits, sequence_length
        




    