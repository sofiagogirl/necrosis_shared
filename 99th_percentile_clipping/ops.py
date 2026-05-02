import json
import glob
import shutil
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization


def copy_code(model_path):
    """Copy all Python source files and relevant directories to the model output folder."""
    tf.io.gfile.mkdir(model_path + 'code')

    for file_name in glob.glob('*.py'):
        try:
            shutil.copy(file_name, model_path + 'code')
        except Exception:
            print("Failed to copy file:", sys.exc_info())

    for dir_name in ['models/', 'helper_scripts/']:
        try:
            shutil.copytree(dir_name, model_path + 'code/' + dir_name)
        except Exception:
            print(f"Skipped copying {dir_name}: source does not exist or target already exists.")


def verbose_msg(text, value, json_format=False):
    """Format a log message from lists of metric names and values."""
    assert len(text) == len(value)

    if json_format:
        msg = json.dumps(
            json.loads(
                json.dumps({t: float(v) for t, v in zip(text, value)}),
                parse_float=lambda x: round(float(x), 4)
            )
        )
    else:
        msg = ""
        for txt, v in zip(text, value):
            msg += f"{txt}: {float(v):.4f} || "

    return msg


def print_and_save_msg(msg, file):
    """Print a message to console and append it to a log file."""
    print(msg)
    with open(file, 'a') as f:
        f.write(msg)


def freeze_model(model, freeze_batch_norm=False):
    """
    Freeze a Keras model's layers.

    Args:
        model: a Keras model
        freeze_batch_norm: if False, batch normalization layers remain trainable
    """
    for layer in model.layers:
        if not freeze_batch_norm and isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
    return model


def normalize(x):
    """Min-max normalize an array to [0, 1]."""
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))