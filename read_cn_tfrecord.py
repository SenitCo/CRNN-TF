#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os 
import numpy as np 
import tensorflow as tf 
import config

def read_tfrecord(tfrecord_dir, num_epochs):
    file_patterns = '*.tfrecord'
    filename = os.path.join(tfrecord_dir, file_patterns)    
    data_files = tf.gfile.Glob(filename)  # 参数可以是文件名或者包含通配符的正则表达式
    filename_queue = tf.train.string_input_producer(data_files, num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature((), dtype=tf.string),
        'label': tf.VarLenFeature(dtype=tf.int64),
        'filename': tf.FixedLenFeature([1], dtype=tf.string)
        })
    images = tf.image.decode_jpeg(features['image'])
    images = tf.image.convert_image_dtype(images, tf.float32)
    images = tf.reshape(images, [config.cfg.IMAGE_HEIGHT, config.cfg.IMAGE_WIDTH, config.cfg.IMAGE_DEPTH])
    labels = tf.cast(features['label'], tf.int32)
    # labels = tf.serialize_sparse(labels)    
    imagenames = features['filename']
    return images, labels, imagenames

def calculate_accuracy(preds, labels):
    accuracy = 0
    total_count = 0
    correct_count = 0
    for index, label in enumerate(labels):
        pred = preds[index]
        pred_length = len(pred)
        label_length = len(label)
        total_count += label_length
        length = label_length if label_length < pred_length else pred_length
        for i in range(length):
            if pred[i] == label[i]:
                correct_count += 1
    if total_count != 0:
        accuracy = correct_count / total_count
    else:
        accuracy = 0
    return accuracy


def sparse_tensor_to_str(sparse_tensor: tf.SparseTensor, lexicon_path):
    if not os.path.exists(lexicon_path):
        raise ValueError("{:s} dostn\'t not exist".format(lexicon_path))
    lexicon = []
    with open(lexicon_path, 'rt', encoding='utf-8') as f:
        for line in f:
            lexicon.append(line)

    indices = sparse_tensor.indices
    values = sparse_tensor.values
    dense_shape = sparse_tensor.dense_shape
    number_list = np.zeros(dense_shape, dtype=values.dtype)
    result = []
    for i, index in enumerate(indices):
        number_list[index[0], index[1]] = values[i]
    for number in number_list:
        result.append(''.join(lexicon[j] for j in number if j != 0))
    return result
