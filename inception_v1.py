# -*- coding: utf-8 -*-
# Code from https://github.com/fchollet/deep-learning-models/pull/59
# Weights can be downloaded from Tensorflow modelzoo
# Adapted to make it compile...
"""Inception V1 model for Keras.

Note that the input preprocessing function is different from the the VGG16
and ResNet models (same as Xception).

Also that (currently) the output predictions are for 1001 classes
(with the 0 class being 'background'), so require a shift compared to the
other models here.

# Reference

- [Going deeper with convolutions](http://arxiv.org/abs/1409.4842v1)

"""
from __future__ import absolute_import, print_function

import numpy as np
from keras import backend as K
from keras import layers
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dropout, Flatten, GlobalAveragePooling2D,
                          GlobalMaxPooling2D, Input, MaxPooling2D)
from keras.models import Model
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import get_source_inputs

WEIGHTS_PATH = 'http://redcatlabs.com/downloads/inception_v1_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = ('http://redcatlabs.com/downloads/'
                       'inception_v1_weights_tf_dim_ordering_tf_kernels_notop.h5')


def conv2d_bn(input_data,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              normalizer=True,
              activation='relu',
              name=None):
    """Utility function to apply conv + BN.
    Arguments:
        input_data: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution, `name + '_bn'` for the
            batch norm layer and `name + '_act'` for the
            activation layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        act_name = name + '_act'
    else:
        conv_name = None
        bn_name = None
        act_name = None
    bn_axis = 3
    input_data = Conv2D(
        filters, (num_row, num_col),
        strides=strides, padding=padding,
        use_bias=False, name=conv_name)(input_data)
    if normalizer:
        input_data = BatchNormalization(
            axis=bn_axis, scale=False, name=bn_name)(input_data)
    if activation:
        input_data = Activation(activation, name=act_name)(input_data)
    return input_data

# Convenience function for 'standard' Inception concatenated blocks


def concatenated_block(input_data, specs, name):
    (br0, br1, br2, br3) = specs   # ((64,), (96,128), (16,32), (32,))

    branch_0 = conv2d_bn(input_data, br0[0], 1, 1, name=name+"_Branch_0_a_1x1")

    branch_1 = conv2d_bn(input_data, br1[0], 1, 1, name=name+"_Branch_1_a_1x1")
    branch_1 = conv2d_bn(branch_1, br1[1], 3, 3, name=name+"_Branch_1_b_3x3")

    branch_2 = conv2d_bn(input_data, br2[0], 1, 1, name=name+"_Branch_2_a_1x1")
    branch_2 = conv2d_bn(branch_2, br2[1], 3, 3, name=name+"_Branch_2_b_3x3")

    branch_3 = MaxPooling2D((3, 3), strides=(
        1, 1), padding='same', name=name+"_Branch_3_a_max")(input_data)
    branch_3 = conv2d_bn(branch_3, br3[0], 1, 1, name=name+"_Branch_3_b_1x1")

    input_data = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=3, name=name+"_Concatenated")
    return input_data


def inception_v1(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1001):
    """Instantiates the Inception v1 architecture.

    This architecture is defined in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/abs/1409.4842v1

    Optionally loads weights pre-trained
    on ImageNet.
    The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 224x224.
    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1001:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1001')

    # Determine proper input shape
    input_shape = (224, 224, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor, shape=input_shape)

    if not K.image_data_format() == 'channels_first':
        raise ValueError('Please specify Keras to use channels_last.')

    # 'Sequential bit at start'
    network_layers = img_input
    network_layers = conv2d_bn(network_layers,  64, 7, 7, strides=(2, 2),
                               padding='same',  name='Conv2d_1a_7x7')

    network_layers = MaxPooling2D((3, 3), strides=(2, 2), padding='same',
                                  name='MaxPool_2a_3x3')(network_layers)

    network_layers = conv2d_bn(network_layers,  64, 1, 1, strides=(1, 1),
                               padding='same', name='Conv2d_2b_1x1')
    network_layers = conv2d_bn(network_layers, 192, 3, 3, strides=(1, 1),
                               padding='same', name='Conv2d_2c_3x3')

    network_layers = MaxPooling2D((3, 3), strides=(2, 2), padding='same',
                                  name='MaxPool_3a_3x3')(network_layers)

    # Now the '3' level inception units
    network_layers = concatenated_block(
        network_layers, ((64,), (96, 128), (16, 32), (32,)), 'Mixed_3b')
    network_layers = concatenated_block(
        network_layers, ((128,), (128, 192), (32, 96), (64,)), 'Mixed_3c')

    network_layers = MaxPooling2D((3, 3), strides=(2, 2), padding='same',
                                  name='MaxPool_4a_3x3')(network_layers)

    # Now the '4' level inception units
    network_layers = concatenated_block(
        network_layers, ((192,), (96, 208), (16, 48), (64,)), 'Mixed_4b')
    network_layers = concatenated_block(
        network_layers, ((160,), (112, 224), (24, 64), (64,)), 'Mixed_4c')
    network_layers = concatenated_block(
        network_layers, ((128,), (128, 256), (24, 64), (64,)), 'Mixed_4d')
    network_layers = concatenated_block(
        network_layers, ((112,), (144, 288), (32, 64), (64,)), 'Mixed_4e')
    network_layers = concatenated_block(
        network_layers, ((256,), (160, 320), (32, 128), (128,)), 'Mixed_4f')

    network_layers = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                  name='MaxPool_5a_2x2')(network_layers)

    # Now the '5' level inception units
    network_layers = concatenated_block(
        network_layers, ((256,), (160, 320), (32, 128), (128,)), 'Mixed_5b')
    network_layers = concatenated_block(
        network_layers, ((384,), (192, 384), (48, 128), (128,)), 'Mixed_5c')

    if include_top:
        # Classification block

        # 'AvgPool_0a_7x7'
        network_layers = AveragePooling2D(
            (7, 7), strides=(1, 1), padding='valid')(network_layers)

        # 'Dropout_0b'
        # slim has keep_prob (@0.8), keras uses drop_fraction
        network_layers = Dropout(0.2)(network_layers)

        # Write out the logits explictly, since it is pretty different
        network_layers = Conv2D(classes, (1, 1), strides=(1, 1),
                                padding='valid', use_bias=True, name='Logits')(network_layers)

        network_layers = Flatten(name='Logits_flat')(network_layers)
        network_layers = Activation(
            'softmax', name='Predictions')(network_layers)
    else:
        if pooling == 'avg':
            network_layers = GlobalAveragePooling2D(
                name='global_pooling')(network_layers)
        elif pooling == 'max':
            network_layers = GlobalMaxPooling2D(
                name='global_pooling')(network_layers)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Finally : Create model
    model = Model(inputs, network_layers, name='inception_v1')
    if weights == 'imagenet':
        weights_path = load_weights(include_top)
        model.load_weights(weights_path)

    return model


def load_weights(include_top):
    if include_top:
        return get_file(
            'inception_v1_weights_tf_dim_ordering_tf_kernels.h5',
            WEIGHTS_PATH,
            cache_subdir='models',
            md5_hash='723bf2f662a5c07db50d28c8d35b626d')
    return get_file(
        'inception_v1_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        md5_hash='6fa8ecdc5f6c402a59909437f0f5c975')


def preprocess_input(input_data):
    input_data /= 255.
    input_data -= 0.5
    input_data *= 2.
    return input_data


if __name__ == '__main__':
    inception_model = inception_v1(include_top=True, weights='imagenet')

    IMG_PATH = 'elephant.jpg'
    img = image.load_img(IMG_PATH, target_size=(224, 224))
    input_image = image.img_to_array(img)
    input_image = np.expand_dims(input_image, axis=0)

    input_image = preprocess_input(input_image)

    preds = inception_model.predict(input_image)

    # Extra shift to remove 'background' as entry0
    preds_1000 = preds[:, 1:]

    print('Predicted:', decode_predictions(preds_1000))
