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
    def __init__(self, scale_range, pre_offset, normalize_label,
                 axis=None, clip_before=False, clip_after=False,
                 *,
                 batch_mode: bool):
        super().__init__("SumScaleNormalizer", axis, clip_before, clip_after)
        self.scale_range = list(scale_range)
        self.pre_offset = float(pre_offset)
        self.normalize_label = normalize_label
        self._min_scale = min(scale_range)
        self._max_scale = max(scale_range)
        self.batch_mode = batch_mode
        self._denorm_pars = {'pre_offset': self.pre_offset}


    def map(self, *args):
        args = list(super().map(*args))
        eps = tf.constant(1e-12, dtype=args[0].dtype)

        x, y = args[0], args[1]

        # --- Achsenwahl ---
        if self.batch_mode and tf.rank(x) == 5:
            # Eingabe-Shape: (B, D, H, W, C)
            reduce_axes = (2, 3, 4)   # normiere pro Sample über HWC
        else:
            # Eingabe-Shape: (D, H, W, C)
            reduce_axes = (1, 2, 3)   # normiere pro Sample über HWC

        # --- Features ---
        x = self._pre_clipping(x + self.pre_offset)
        scale = tf.random.uniform((1,), minval=self._min_scale, maxval=self._max_scale)
        sum_feature = tf.maximum(tf.reduce_sum(x, axis=reduce_axes, keepdims=True), eps)
        x = self._post_clipping(x / sum_feature * scale)

        # --- Labels ---
        if self.normalize_label:
            y = self._pre_clipping(y + self.pre_offset)
            sum_label = tf.maximum(tf.reduce_sum(y, axis=reduce_axes, keepdims=True), eps)
            y = self._post_clipping(y / sum_label * scale)

        self._denorm_pars['scale'] = scale
        self._denorm_pars['sum'] = sum_feature
        return (x, y)
