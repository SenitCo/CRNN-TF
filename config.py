#!/usr/bin/env python
# -*- coding:utf-8 -*-

from easydict import EasyDict 
__C = EasyDict()

cfg = __C

# The number of hidden units of rnn
__C.NUM_UNITS = 256
# The number of classes
__C.NUM_CLASSES = 5990

# image width
__C.IMAGE_WIDTH = 280
# image height
__C.IMAGE_HEIGHT = 32
# image depth
__C.IMAGE_DEPTH = 3

# Training options
__C.TRAIN = EasyDict()
# training epochs
__C.TRAIN.MAX_ITER_STEPS = 100000
# the display step
__C.TRAIN.DISPLAY_STEP = 1
# the validation step
__C.TRAIN.VAL_DIASPLAY_STEP = 100
# the step the model saved
__C.TRAIN.SAVE_STEP = 5000
# the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.05
# whether allow the GPU growth during training process
__C.TRAIN.TF_ALLOW_GROWTH = True
# training batch size
__C.TRAIN.BATCH_SIZE = 32
# validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 32
# the learning rate decay rate
__C.TRAIN.LR_DECAY_RATE = 0.5
# the learning rate decay step
__C.TRAIN.LR_DECAY_STEPS = 20000


# Test options
__C.TEST = EasyDict()
__C.TEST.BATCH_SIZE = 32