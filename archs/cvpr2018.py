# cvpr2018.py ---
#
# Filename: cvpr2018.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Oct  9 17:35:01 2017 (+0200)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C)
# Visual Computing Group @ University of Victoria
# Computer Vision Lab @ EPFL

# Code:

import tensorflow as tf

from ..ops import conv1d_layer, conv1d_resnet_block


def build_graph(x_in, is_training, config):

    activation_fn = tf.nn.relu

    x_in_shp = tf.shape(x_in)

    cur_input = x_in
    print(cur_input.shape)
    idx_layer = 0
    numlayer = config.net_depth
    ksize = 1
    nchannel = config.net_nchannel
    # Use resnet or simle net
    act_pos = config.net_act_pos
    conv1d_block = conv1d_resnet_block

    # First convolution
    with tf.variable_scope("hidden-input"):
        cur_input = conv1d_layer(
            inputs=cur_input,
            ksize=1,
            nchannel=nchannel,
            activation_fn=None,
            perform_bn=False,
            perform_gcn=False,
            is_training=is_training,
            act_pos="pre",
            data_format="NHWC",
        )
        print(cur_input.shape)
    for _ksize, _nchannel in zip(
            [ksize] * numlayer, [nchannel] * numlayer):
        scope_name = "hidden-" + str(idx_layer)
        with tf.variable_scope(scope_name):
            cur_input = conv1d_block(
                inputs=cur_input,
                ksize=_ksize,
                nchannel=_nchannel,
                activation_fn=activation_fn,
                is_training=is_training,
                perform_bn=config.net_batchnorm,
                perform_gcn=config.net_gcnorm,
                act_pos=act_pos,
                data_format="NHWC",
            )
            # Apply pooling if needed
            print(cur_input.shape)

        idx_layer += 1

    with tf.variable_scope("output"):
        cur_input = conv1d_layer(
            inputs=cur_input,
            ksize=1,
            nchannel=1,
            activation_fn=None,
            is_training=is_training,
            perform_bn=False,
            perform_gcn=False,
            data_format="NHWC",
        )
        #  Flatten
        cur_input = tf.reshape(cur_input, (x_in_shp[0], x_in_shp[2]))

    logits = cur_input
    print(cur_input.shape)

    return logits


#
# cvpr2018.py ends here
