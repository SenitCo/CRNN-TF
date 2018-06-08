#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np 
import tensorflow as tf 
import argparse
import time
import crnn_model
import read_cn_tfrecord
import config

root_dir = "/home/shenzhao/Datasets"
dataset_dir = root_dir + "/SynChiData"
tfrecord_dir = dataset_dir + "/tfrecord"
model_dir = "./model"
lexicon_path = dataset_dir + '/char_std_5990.txt'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=tfrecord_dir, help='Where the data stored?')
    parser.add_argument('--weight_dir', type=str, default=model_dir, help='Where the model saved?')
    return parser.parse_args()

def train_crnn(data_dir, weight_dir):
    train_data_dir = data_dir + "/train"
    batch_size = config.cfg.TRAIN.BATCH_SIZE
    images, labels, imagenames = read_cn_tfrecord.read_tfrecord(train_data_dir, num_epochs=None)
    input_images, input_labels = tf.train.shuffle_batch(tensors=[images, labels], batch_size=batch_size, capacity=50000, min_after_dequeue=1000, num_threads=4)
    crnn_logits, sequence_length = crnn_model.build_crnn(input_images, training=True, num_units=config.cfg.NUM_UNITS, num_classes=config.cfg.NUM_CLASSES)
    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels, inputs=crnn_logits, sequence_length=sequence_length))
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(crnn_logits, sequence_length=sequence_length, merge_repeated=False)
    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))

    global_step = tf.Variable(0, name='global_step', trainable=False)
    init_learning_rate = config.cfg.TRAIN.LEARNING_RATE
    lr_decay_rate = config.cfg.TRAIN.LR_DECAY_RATE 
    lr_decay_steps = config.cfg.TRAIN.LR_DECAY_STEPS
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, lr_decay_steps, lr_decay_rate,staircase=True)
    
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)

    tf.summary.scalar(name='Cost', tensor=cost)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Sequence_Dist', tensor=sequence_dist)
    merge_summary_op = tf.summary.merge_all()

    tfboard_path = './tfboard'
    if not os.path.exists(tfboard_path):
        os.makedirs(tfboard_path)
    summary_writer = tf.summary.FileWriter(tfboard_path)

    max_iter_steps = config.cfg.TRAIN.MAX_ITER_STEPS
    display_step = config.cfg.TRAIN.DISPLAY_STEP
    val_step = config.cfg.TRAIN.VAL_DIASPLAY_STEP
    save_step = config.cfg.TRAIN.SAVE_STEP 

    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(weight_dir)
        if ckpt and ckpt.model_checkpoint_path:
            index = ckpt.model_checkpoint_path.split('-')[-1]
            start = int(index)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("\033[0;32;40m\tRestore model from {:s}\033[0m".format(weight_dir))
        else:
            print("\033[0;32;40m\tTraining from scratch\033[0m")
            start = 0 

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        accuracy = []
        for step in range(start, max_iter_steps):
            if coord.should_stop():
                break
            train_op, loss, summary = sess.run([optimizer, cost, merge_summary_op]) 
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            print("Step: {:d} cost: {:9f}".format(step+1, loss))
            if step % val_step == 0:
                label_seq, pred_seq, seq_dist = sess.run([input_labels, decoded, sequence_dist]) 
                pred_seq = read_cn_tfrecord.sparse_tensor_to_str(pred_seq[0], lexicon_path)
                label_seq = read_cn_tfrecord.sparse_tensor_to_str(label_seq, lexicon_path)
                accuracy = read_cn_tfrecord.calculate_accuracy(pred_seq, label_seq)
                
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                print("Step: {:d} cost: {:9f} sequence distance: {:9f}, train accuracy: {:9f}".format(step+1, loss, seq_dist, accuracy))
    
            if step % save_step == 0:
                saver.save(sess=sess, save_path=os.path.join(weight_dir, 'model.ckpt'), global_step=global_step)
                
            summary_writer.add_graph(sess.graph)
            summary_writer.add_summary(summary=summary, global_step=step)
        coord.request_stop()
        coord.join(threads=threads)
    sess.close()

def main():
    args = init_args()

    if not os.path.exists(args.data_dir):
        raise ValueError("{:s} doesn\'t exist".format(args.data_dir))

    if not os.path.exists(args.weight_dir):
        os.makedirs(args.weight_dir)
    
    train_crnn(args.data_dir, args.weight_dir)
    print("Done!\n")

if __name__ == '__main__':
    main()



