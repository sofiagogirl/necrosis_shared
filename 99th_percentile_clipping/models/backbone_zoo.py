from __future__ import absolute_import

import warnings

from tensorflow.keras.applications import *
from tensorflow.keras.models import Model

from ops import freeze_model


# ======================= backbone layer map =====================

layer_candidates = {
    'VGG16':         ('block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'),
    'VGG19':         ('block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4'),
    'ResNet50':      ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'),
    'ResNet101':     ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'),
    'ResNet152':     ('conv1_relu', 'conv2_block3_out', 'conv3_block8_out', 'conv4_block36_out', 'conv5_block3_out'),
    'ResNet50V2':    ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu', 'post_relu'),
    'ResNet101V2':   ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block23_1_relu', 'post_relu'),
    'ResNet152V2':   ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block8_1_relu', 'conv4_block36_1_relu', 'post_relu'),
    'DenseNet121':   ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet169':   ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet201':   ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'EfficientNetB0': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB1': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB2': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB3': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB4': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB5': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB6': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB7': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
}


# ======================= backbone utils =========================

def bach_norm_checker(backbone_name, batch_norm):
    """Warn if batch norm setting is inconsistent with the chosen backbone."""
    batch_norm_backbone = 'VGG' not in backbone_name

    if batch_norm_backbone != batch_norm:
        if batch_norm_backbone:
            msg = f"\n\nBackbone {backbone_name} uses batch norm, but other layers received batch_norm={batch_norm}"
        else:
            msg = f"\n\nBackbone {backbone_name} does not use batch norm, but other layers received batch_norm={batch_norm}"
        warnings.warn(msg)


def backbone_zoo(backbone_name, weights, input_tensor, depth, freeze_backbone, freeze_batch_norm):
    """
    Configure a backbone encoder from tensorflow.keras.applications.

    Supported backbones: VGG16/19, ResNet50/101/152, ResNet50V2/101V2/152V2,
    DenseNet121/169/201, EfficientNetB[0-7].

    Args:
        backbone_name: name of the backbone from tensorflow.keras.applications.
        weights: None, 'imagenet', or path to weights file.
        input_tensor: input tensor.
        depth: number of encoded feature maps (e.g. depth=4 for four downsampling levels).
        freeze_backbone: True to freeze backbone weights.
        freeze_batch_norm: False to keep batch normalization layers trainable.

    Returns:
        model: a Keras backbone model.
    """
    candidates = layer_candidates[backbone_name]
    depth = min(depth, len(candidates))

    backbone_func = eval(backbone_name)
    backbone_ = backbone_func(include_top=False, weights=weights,
                              input_tensor=input_tensor, pooling=None)

    X_skip = [backbone_.get_layer(candidates[i]).output for i in range(depth)]

    model = Model(inputs=[input_tensor], outputs=X_skip,
                  name='{}_backbone'.format(backbone_name))

    if freeze_backbone:
        model = freeze_model(model, freeze_batch_norm=freeze_batch_norm)

    return model