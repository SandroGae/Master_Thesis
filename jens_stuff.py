# jens_stuff.py
#!/usr/bin/env python3
"""
Single-file version of Jens' utilities:
Includes DatasetGenerator, SumScaleNormalizer, and required utils.
"""

import os
import h5py
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from abc import ABC, abstractmethod


# ============ UTILS ============

def get_timestamp():
    return datetime.now().strftime('%Y/%m/%d %H:%M:%S')

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)



# ============ PRE-PROCESSING BASE CLASSES ============

class DataProcessor(ABC):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        attributes = ", ".join(f"{k}={v}" for k,v in vars(self).items() if not k.startswith("_"))
        return f"{self.__class__.__name__}({attributes})"

    def get_config(self):
        attributes = self.__dict__.keys()
        return {attr: getattr(self, attr) for attr in attributes if not attr.startswith("_")}

    @abstractmethod
    def map(self, *args):
        return [tf.cast(arg, tf.float32) for arg in args]


class Normalizer(DataProcessor):
    def __init__(self, name, axis=None, clip_before=False, clip_after=False):
        super().__init__(name)
        self.axis = axis
        self.clip_before = list(clip_before) if clip_before is not False else False
        self.clip_after = list(clip_after) if clip_after is not False else False
        self._denorm_pars = {}

    def _pre_clipping(self, tensor):
        if self.clip_before is not False:
            return tf.clip_by_value(tensor, self.clip_before[0], self.clip_before[1])
        return tensor

    def _post_clipping(self, tensor):
        if self.clip_after is not False:
            return tf.clip_by_value(tensor, self.clip_after[0], self.clip_after[1])
        return tensor

    def map(self, *args):
        return super().map(*args)

    def inverse_map(self, tensor, length):
        if len(self._denorm_pars) < length:
            raise Exception(f"{get_timestamp()} - NORMALIZER: Missing denorm parameters.")
        return tf.cast(tensor, tf.float32)


# ============ SUM SCALE NORMALIZER (used by Jens) ============

class SumScaleNormalizer(Normalizer):
    """
    Divide by sum, then multiply with random scale factor.
    """

    def __init__(self, scale_range, pre_offset, normalize_label, axis=None,
                 batch_mode=False, clip_before=False, clip_after=False):
        super().__init__("SumScaleNormalizer", axis, clip_before, clip_after)
        self.scale_range = list(scale_range)
        self.pre_offset = float(pre_offset)
        self.normalize_label = normalize_label
        self._batch_mode = batch_mode
        self._min_scale = min(scale_range)
        self._max_scale = max(scale_range)
        self._denorm_pars = {'pre_offset': self.pre_offset}

    def map(self, *args):
        args = super().map(*args)
        # Features
        args[0] = self._pre_clipping(args[0] + self.pre_offset)
        scale = tf.random.uniform(
            shape=(args[0].shape[0] if self._batch_mode else 1,),
            minval=self._min_scale,
            maxval=self._max_scale
        )
        sum_feature = tf.math.reduce_sum(args[0], axis=self.axis, keepdims=True)
        args[0] = args[0] / sum_feature * scale
        args[0] = self._post_clipping(args[0])

        if self.normalize_label:
            args[1] = self._pre_clipping(args[1] + self.pre_offset)
            args[1] = args[1] / tf.math.reduce_sum(args[1], axis=self.axis, keepdims=True) * scale
            args[1] = self._post_clipping(args[1])

        self._denorm_pars['scale'] = scale
        self._denorm_pars['sum'] = sum_feature
        return args

    def inverse_map(self, tensor):
        tensor = super().inverse_map(tensor, length=3)
        return tensor / self._denorm_pars['scale'] * self._denorm_pars['sum'] - self._denorm_pars['pre_offset']


# ============ DATASET GENERATOR ============

class DatasetGenerator:
    """
    Simplified version of Jens' DatasetGenerator.
    Works with in-memory features & labels and optional preprocessor.
    """

    def __init__(self, preprocessor=None, augmenter=None,
                 features=None, labels=None, weights=None):
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        self._features = features
        self._labels = labels
        self._weights = weights
        self._provided_tensors = True

        # Define mapping funcs
        self._pp_map = (lambda *args: self.preprocessor.map(*args)) if self.preprocessor else None
        self._aug_map = (lambda *args: self.augmenter.map(*args)) if self.augmenter else None

    def create_dataset(self, input_shape, batch_size, seed, shuffle=True):
        if input_shape != self._features.shape[1:]:
            raise Exception(f"Input shape {input_shape} does not match features {self._features.shape[1:]}")

        if self._weights is not None:
            dataset = tf.data.Dataset.from_tensor_slices((self._features, self._labels, self._weights))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self._features, self._labels))

        if self._pp_map is not None:
            dataset = dataset.map(self._pp_map, num_parallel_calls=tf.data.AUTOTUNE).cache()

        if self._aug_map is not None:
            dataset = dataset.map(self._aug_map, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self._features.shape[0],
                                      reshuffle_each_iteration=True,
                                      seed=seed)

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
