# ops.py ---
#
# Filename: ops.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Tue Apr  3 14:09:17 2018 (-0700)
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

import numpy as np

from six.moves import xrange


# ------------------------------------------------------------
# Tensorflow ops

def tf_get_shape_as_list(x):

    return [_s if _s is not None else - 1 for _s in x.get_shape().as_list()]


def tf_quaternion_from_matrix(M):

    import tensorflow as tf

    m00 = M[:, 0, 0][..., None]
    m01 = M[:, 0, 1][..., None]
    m02 = M[:, 0, 2][..., None]
    m10 = M[:, 1, 0][..., None]
    m11 = M[:, 1, 1][..., None]
    m12 = M[:, 1, 2][..., None]
    m20 = M[:, 2, 0][..., None]
    m21 = M[:, 2, 1][..., None]
    m22 = M[:, 2, 2][..., None]
    # symmetric matrix K
    zeros = tf.zeros_like(m00)
    K = tf.concat(
        [m00 - m11 - m22, zeros, zeros, zeros,
         m01 + m10, m11 - m00 - m22, zeros, zeros,
         m02 + m20, m12 + m21, m22 - m00 - m11, zeros,
         m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        axis=1)
    K = tf.reshape(K, (-1, 4, 4))
    K /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = tf.self_adjoint_eig(K)

    q0 = V[:, 3, 3][..., None]
    q1 = V[:, 0, 3][..., None]
    q2 = V[:, 1, 3][..., None]
    q3 = V[:, 2, 3][..., None]
    q = tf.concat([q0, q1, q2, q3], axis=1)
    sel = tf.reshape(tf.to_float(q[:, 0] < 0.0), (-1, 1))
    q = (1.0 - sel) * q - sel * q

    return q


def tf_matrix_from_quaternion(q, eps=1e-10):

    import tensorflow as tf

    # Make unit quaternion
    q_norm = q / (eps + tf.norm(q, axis=1, keep_dims=True))
    q_norm *= tf.constant(2.0 ** 0.5, dtype=tf.float32)
    qq = tf.matmul(
        tf.reshape(q_norm, (-1, 4, 1)),
        tf.reshape(q_norm, (-1, 1, 4))
    )
    M = tf.stack([
        1.0 - qq[:, 2, 2] - qq[:, 3, 3], qq[:, 1, 2] - qq[:, 3, 0],
        qq[:, 1, 3] + qq[:, 2, 0], qq[:, 1, 2] + qq[:, 3, 0],
        1.0 - qq[:, 1, 1] - qq[:, 3, 3], qq[:, 2, 3] - qq[:, 1, 0],
        qq[:, 1, 3] - qq[:, 2, 0], qq[:, 2, 3] + qq[:, 1, 0],
        1.0 - qq[:, 1, 1] - qq[:, 2, 2]
    ], axis=1)

    return M


def tf_skew_symmetric(v):

    import tensorflow as tf

    zero = tf.zeros_like(v[:, 0])

    M = tf.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M


def tf_unskew_symmetric(M):

    import tensorflow as tf

    v = tf.stack([
        0.5 * (M[:, 7] - M[:, 5]),
        0.5 * (M[:, 2] - M[:, 6]),
        0.5 * (M[:, 3] - M[:, 1]),
    ], axis=1)

    return v


# ------------------------------------------------------------
# Architecture related

def bn_act(linout, perform_gcn, perform_bn, activation_fn, is_training,
           data_format):

    import tensorflow as tf

    """ Perform batch normalization and activation """
    if data_format == "NHWC":
        axis = -1
    else:
        axis = 1

    # Global Context normalization on the input
    if perform_gcn:
        # Epsilon to be used in the tf.nn.batch_normalization
        var_eps = 1e-3
        # get mean variance for single sample (channel-wise, note that we omit
        # axis=1 since we are expecting a size of 1 in that dimension)
        mean, variance = tf.nn.moments(linout, axes=[2], keep_dims=True)
        # Use tensorflow's nn.batchnorm
        linout = tf.nn.batch_normalization(
            linout, mean, variance, None, None, var_eps)

    if perform_bn:
        with tf.variable_scope("bn"):
            linout = tf.layers.batch_normalization(
                inputs=linout,
                center=False, scale=False,
                training=is_training,
                trainable=True,
                axis=axis,
            )

    if activation_fn is None:
        output = linout
    else:
        output = activation_fn(linout)

    return output


def pad_cyclic(tensor, paddings):

    import tensorflow as tf

    ndim = len(paddings)
    for _dim, _pad in zip(xrange(ndim), paddings):

        pad_list = []
        if _pad[0] > 0:
            # Padding to put at front
            slice_st = [slice(None, None)] * ndim
            slice_st[_dim] = slice(-_pad[0], None)
            pad_list += [tensor[tuple(slice_st)]]

        # Original
        pad_list += [tensor]

        if _pad[1] > 0:
            # Padding to put at back
            slice_ed = [slice(None, None)] * ndim
            slice_ed[_dim] = slice(None, _pad[1])
            pad_list += [tensor[tuple(slice_ed)]]

        if len(pad_list) > 1:
            # Concatenate to do padding
            tensor = tf.concat(pad_list, axis=_dim)

    return tensor


def conv1d_pad_cyclic(inputs, ksize, numconv, data_format="NCHW"):
    in_shp = tf_get_shape_as_list(inputs)
    ksize = 2 * (ksize // 2 * numconv) + 1

    if data_format == "NCHW":
        assert (ksize < in_shp[-1]) or (in_shp[-1] == -1)
        if np.mod(ksize, 2) == 0:
            paddings = [
                [0, 0], [0, 0], [0, 0], [ksize // 2 - 1, ksize // 2]
            ]
        else:
            paddings = [
                [0, 0], [0, 0], [0, 0], [ksize // 2, ksize // 2]
            ]
    else:
        assert (ksize < in_shp[-2]) or (in_shp[-2] == -1)
        if np.mod(ksize, 2) == 0:
            paddings = [
                [0, 0], [0, 0], [ksize // 2 - 1, ksize // 2], [0, 0]
            ]
        else:
            paddings = [
                [0, 0], [0, 0], [ksize // 2, ksize // 2], [0, 0]
            ]
    inputs = pad_cyclic(inputs, paddings)

    return inputs


def get_W_b_conv1d(in_channel, out_channel, ksize, dtype=None):

    import tensorflow as tf

    if dtype is None:
        dtype = tf.float32

    fanin = in_channel * ksize
    W = tf.get_variable(
        "weights", shape=[1, ksize, in_channel, out_channel], dtype=dtype,
        initializer=tf.truncated_normal_initializer(stddev=2.0 / fanin),
        # initializer=tf.random_normal_initializer(stddev=0.02),
    )
    b = tf.get_variable(
        "biases", shape=[out_channel], dtype=dtype,
        initializer=tf.zeros_initializer(),
    )
    # tf.summary.histogram("W", W)
    # tf.summary.histogram("b", b)

    return W, b


def conv1d_layer(inputs, ksize, nchannel, activation_fn, perform_bn,
                 perform_gcn, is_training, perform_kron=False,
                 padding="CYCLIC", data_format="NCHW",
                 act_pos="post"):

    import tensorflow as tf

    assert act_pos == "pre" or act_pos == "post"

    # Pad manually
    if padding == "CYCLIC":
        if ksize > 1:
            inputs = conv1d_pad_cyclic(
                inputs, ksize, 1, data_format=data_format)
        cur_padding = "VALID"
    else:
        cur_padding = padding

    in_shp = tf_get_shape_as_list(inputs)
    if data_format == "NHWC":
        in_channel = in_shp[-1]
        ksizes = [1, 1, ksize, 1]
    else:
        in_channel = in_shp[1]
        ksizes = [1, 1, 1, ksize]

    assert len(in_shp) == 4

    # # Lift with kronecker
    # if not is_first:
    #     inputs = tf.concat([
    #         inputs,
    #         kronecker_layer(inputs),
    #     ], axis=-1)

    pool_func = None
    self_ksize = ksize
    do_add = False

    # If pre activation
    if act_pos == "pre":
        inputs = bn_act(inputs, perform_gcn, perform_bn, activation_fn,
                        is_training, data_format)

    # Normal convolution
    with tf.variable_scope("self-conv"):
        W, b = get_W_b_conv1d(in_channel, nchannel, self_ksize)
        # Convolution in the valid region only
        linout = tf.nn.conv2d(
            inputs, W, [1, 1, 1, 1], cur_padding, data_format=data_format)
        linout = tf.nn.bias_add(linout, b, data_format=data_format)
    # Pooling Convolution for the summary route
    if pool_func is not None:
        with tf.variable_scope("neigh-conv"):
            if not do_add:
                linout = pool_func(
                    linout,
                    ksize=ksizes,
                    strides=[1, 1, 1, 1],
                    padding=cur_padding,
                    data_format=data_format,
                )
            else:
                W_n, b_n = get_W_b_conv1d(in_channel, nchannel, 1)
                # Convolution in the valid region only
                linout_n = tf.nn.conv2d(
                    inputs, W_n, [1, 1, 1, 1], "VALID", data_format=data_format
                )
                linout_n = tf.nn.bias_add(
                    linout_n, b_n, data_format=data_format)
                linout_n = pool_func(
                    linout_n,
                    ksize=ksizes,
                    strides=[1, 1, 1, 1],
                    padding=cur_padding,
                    data_format=data_format,
                )
                # # Crop original linout
                # if ksize > 1:
                #     if np.mod(ksize, 2) == 0:
                #         crop_st = ksize // 2 - 1
                #     else:
                #         crop_st = ksize // 2
                #         crop_ed = ksize // 2
                #     linout = linout[:, :, :, crop_st:-crop_ed]
                # Add to the original output
                linout = linout + linout_n

    # If post activation
    output = linout
    if act_pos == "post":
        output = bn_act(linout, perform_gcn, perform_bn, activation_fn,
                        is_training, data_format)

    return output


def conv1d_resnet_block(inputs, ksize, nchannel, activation_fn, is_training,
                        midchannel=None, perform_bn=False, perform_gcn=False,
                        padding="CYCLIC", act_pos="post", data_format="NCHW"):

    import tensorflow as tf

    # In case we want to do a bottleneck layer
    if midchannel is None:
        midchannel = nchannel

    # don't activate conv1 in case of midact
    conv1_act_fn = activation_fn
    if act_pos == "mid":
        conv1_act_fn = None
        act_pos = "pre"

    # Pass branch
    with tf.variable_scope("pass-branch"):
        # passthrough to be used when num_outputs != num_inputs
        in_shp = tf_get_shape_as_list(inputs)
        if data_format == "NHWC":
            in_channel = in_shp[-1]
        else:
            in_channel = in_shp[1]
        if in_channel != nchannel:
            cur_in = inputs
            # Simply change channels through 1x1 conv
            with tf.variable_scope("conv"):
                cur_in = conv1d_layer(
                    inputs=inputs, ksize=1,
                    nchannel=nchannel,
                    activation_fn=None,
                    perform_bn=False,
                    perform_gcn=False,
                    is_training=is_training,
                    padding=padding,
                    data_format=data_format,
                )
            orig_inputs = cur_in
        else:
            orig_inputs = inputs

    # Conv branch
    with tf.variable_scope("conv-branch"):
        cur_in = inputs
        # Do bottle neck if necessary (Linear)
        if midchannel != nchannel:
            with tf.variable_scope("preconv"):
                cur_in = conv1d_layer(
                    inputs=cur_in, ksize=1,
                    nchannel=nchannel,
                    activation_fn=None,
                    perform_bn=False,
                    perform_gcn=False,
                    is_training=is_training,
                    padding=padding,
                    data_format=data_format,
                )
                cur_in = activation_fn(cur_in)

        # Main convolution
        with tf.variable_scope("conv1"):
            # right branch
            cur_in = conv1d_layer(
                inputs=cur_in, ksize=ksize,
                nchannel=nchannel,
                activation_fn=conv1_act_fn,
                perform_bn=perform_bn,
                perform_gcn=perform_gcn,
                is_training=is_training,
                padding=padding,
                act_pos=act_pos,
                data_format=data_format,
            )

        # Main convolution
        with tf.variable_scope("conv2"):
            # right branch
            cur_in = conv1d_layer(
                inputs=cur_in, ksize=ksize,
                nchannel=nchannel,
                activation_fn=activation_fn,
                perform_bn=perform_bn,
                perform_gcn=perform_gcn,
                is_training=is_training,
                padding=padding,
                act_pos=act_pos,
                data_format=data_format,
            )

        # Do bottle neck if necessary (Linear)
        if midchannel != nchannel:
            with tf.variable_scope("postconv"):
                cur_in = conv1d_layer(
                    inputs=cur_in, ksize=1,
                    nchannel=nchannel,
                    activation_fn=None,
                    perform_bn=False,
                    perform_gcn=False,
                    is_training=is_training,
                    padding=padding,
                    data_format=data_format,
                )
                cur_in = activation_fn(cur_in)

    # Crop lb or rb accordingly
    if padding == "VALID" and ksize > 1:
        # Crop pass branch results
        if np.mod(ksize, 2) == 0:
            crop_st = ksize // 2 - 1
        else:
            crop_st = ksize // 2
            crop_ed = ksize // 2
            if data_format == "NHWC":
                orig_inputs = orig_inputs[:, :,  crop_st:-crop_ed, :]
            else:
                orig_inputs = orig_inputs[:, :, :, crop_st:-crop_ed]

    return cur_in + orig_inputs

#
# ops.py ends here
