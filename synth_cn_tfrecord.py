#!/usr/bin/env python
# -*- coding : utf-8 -*-
import os 
import math
import numpy as np
import tensorflow as tf 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

root_dir = "/home/shenzhao/Datasets"
dataset_dir = root_dir + "/SynChiData"
image_dir = dataset_dir + "/images"
tfrecord_dir = dataset_dir + "/tfrecord"
train_list_filename = "train.txt"
test_list_filename = "test.txt"

# 划分训练集和验证集，图像文件名在同一个txt文件中
def split_train_val(dataset_dir, image_list_filename, split_percent=10):
    image_filename, label_seq = get_image_label(dataset_dir, image_list_filename)
    data_length = len(image_filename)
    index = np.random.permutation(data_length)  # 产生随机索引
    val_length = data_length // split_percent
    train = {}
    val = {}
    val['image_filename'] = image_filename[index[:val_length]]
    val['label_seq'] = label_seq[index[val_length]]
    train['image_filename'] = image_filename[index[val_length:]]
    train['label_seq'] = label_seq[index[val_length:]]
    return train, val

# 获取图像文件名和对应的标签序列
def get_image_label(dataset_dir, image_list_filename):
    image_list_filepath = os.path.join(dataset_dir, image_list_filename)
    image_filename = []
    label_seq = []
    with open(image_list_filepath, 'r') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            image_filename.append(line[0])
            seq = [int(i) for i in line[1:]]
            label_seq.append(seq)
    return np.array(image_filename), np.array(label_seq)

# 生成多个TFRecord数据文件，num_shards为文件数
def generate_tfrecord(image_dir, image_filename, label_seq, tfrecord_dir, num_shards=1000, start_shard=0):
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    sess = tf.Session(config=session_config)

    num_digits = math.ceil(math.log10(num_shards - 1))
    shard_format = '%0'+ ('%d' % num_digits) + 'd' # 位数对齐
    data_length = len(image_filename)
    images_per_shard = int(math.ceil(data_length / float(num_shards)))

    for i in range(start_shard, num_shards):
        start = i * images_per_shard
        end = (i + 1) * images_per_shard
        tfrecord_file = tfrecord_dir + "words-" + (shard_format % i) + '.tfrecord'
        if os.path.isfile(tfrecord_file): # Don't recreate data if restarting
            continue
        print(str(i), 'of', str(num_shards), '[', str(start), ':', str(end), ']', tfrecord_file)
        generate_shard(sess, image_dir, image_filename[start : end], label_seq[start : end], tfrecord_file)

    # 处理剩下的数据，作为最后一个shard
    start = num_shards * images_per_shard
    tfrecord_file = tfrecord_dir + "words-" + (shard_format % num_shards) + '.tfrecord'
    print(str(i), 'of', str(num_shards), '[', str(start), ':', str(data_length), ']', tfrecord_file)
    generate_shard(sess, image_dir, image_filename[start: ], label_seq[start: ], tfrecord_file)
    sess.close()



# 将批量图像数据生成单个TFRecord数据文件
def generate_shard(sess, image_dir, image_filename, label_seq, tfrecord_file):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    data_length = len(image_filename)
    for i in range(data_length):
        filename = image_filename[i]
        filepath = os.path.join(image_dir, filename)

        if os.stat(filepath).st_size == 0:  # 文件为空则跳过
            print('SKIPPING', filename)
        try:
            image_data = get_image(sess, filepath)
            label_data = label_seq[i]
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': bytes_feature(tf.compat.as_bytes(image_data)),
                'label': int64_feature(label_data),
                'filename': bytes_feature(tf.compat.as_bytes(filename))
            }))
            writer.write(example.SerializeToString())
        except:
            print('ERROR', filepath)
    writer.close()

# 根据文件路径获取图像数据
def get_image(sess, filepath):
    # image_data = tf.placeholder(dtype=tf.string)
    # image_decoder = tf.image.decode_jpeg(image_data, channels=1)
    with tf.gfile.FastGFile(filepath, 'rb') as f:
        data = f.read()
    # image = sess.run(image_decoder, feed_dict={image_data : data})
    return data

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def main():
    train, val = split_train_val(dataset_dir, train_list_filename)
    test_image_filename, test_label_seq = get_image_label(dataset_dir, test_list_filename)
    generate_tfrecord(image_dir, train['image_filename'], train['label_seq'], tfrecord_dir + '/train/')
    generate_tfrecord(image_dir, val['image_filename'], train['label_seq'], tfrecord_dir + '/val/')
    generate_tfrecord(image_dir, test_image_filename, test_label_seq, tfrecord_dir + '/test/')

if __name__ == '__main__':
    main()