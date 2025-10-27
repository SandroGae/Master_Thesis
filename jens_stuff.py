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

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

# SUM SCALE NORMALIZER (used by Jens)
class SumScaleNormalizer:
    """
    Normiert pro Sample (und pro Slice):
      1) negative Werte entfernen (pre-clip)
      2) durch Summe über (H, W, C) teilen
      3) mit Zufallsfaktor hochskalieren
      4) final auf [0, 1] clippen
    """

    def __init__(self, scale_min=5000.0, scale_max=15000.0, pre_offset=0.0, normalize_label=True, batch_mode=True):
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.pre_offset = float(pre_offset)
        self.normalize_label = bool(normalize_label)
        self.batch_mode = bool(batch_mode)

    def _finite_clip01(self, t):
        t = tf.cast(t, tf.float32) # Einheitlicher dtype
        t = tf.where(tf.math.is_finite(t), t, 0.0) # NaN/Inf -> 0
        return tf.clip_by_value(t, 0.0, 1.0) # [0,1]

    def _reduce_axes(self, x):
        if self.batch_mode == True: # 5D (B,D,H,W,C) -> (H,W,C)
            return (2, 3, 4)
        else:
            return (1, 2, 3)        # 4D (D,H,W,C)-> (H,W,C)

    def map(self, x, y):
        eps = tf.constant(1e-12, tf.float32)
        x = tf.cast(x, tf.float32); y = tf.cast(y, tf.float32)

        x_pc = tf.maximum(x + self.pre_offset, 0.0)
        if self.normalize_label:
            y_pc = tf.maximum(y + self.pre_offset, 0.0)

        axes = (2,3,4) if self.batch_mode else (1,2,3)
        x_sum = tf.maximum(tf.reduce_sum(x_pc, axis=axes, keepdims=True), eps)
        x_scale = tf.cast(tf.random.uniform((1,), minval=self.scale_min, maxval=self.scale_max, dtype=tf.int32), tf.float32)
        x_norm = self._finite_clip01((x_pc / x_sum) * x_scale)

        if self.normalize_label:
            y_sum = tf.maximum(tf.reduce_sum(y_pc, axis=axes, keepdims=True), eps)
            y_scale = tf.cast(tf.random.uniform((1,), minval=self.scale_min, maxval=self.scale_max, dtype=tf.int32), tf.float32)
            y_norm = self._finite_clip01((y_pc / y_sum) * y_scale)
        else:
            y_norm = y

        return x_norm, y_norm

