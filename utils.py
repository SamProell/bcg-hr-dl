import os

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
import tensorflow as tf


def get_group_splitter(n_splits, groups):
    """Get splitter for grouped k-fold cross validation

    If number of splits is equal to the number of groups, leave-one-out
    splitter is used

    Args:
        n_splits (int): number of splits
        groups (`1d array-like`): group list of corresponding data

    Returns:
        `generator`: index generator with train and test indices
    """

    if n_splits == len(np.unique(groups)):
        return LeaveOneGroupOut().split(groups, groups=groups)

    return GroupKFold(n_splits=n_splits).split(groups, groups=groups)


def get_model_from_json(modelpath, filename="model.json"):
    """Read model architecture from JSON and return
    keras model
    
    Args:
        modelpath (str): path to model file
        filename (str): filename of model file
    
    Returns:
        `keras model`: uninitialized keras model
    """

    with open(os.path.join(modelpath, filename), "r") as fp:
        model_json = fp.read()
    return tf.keras.models.model_from_json(model_json)
