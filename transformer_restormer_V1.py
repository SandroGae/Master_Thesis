#!/usr/bin/env python3

import os
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# Deine Custom-Module (wie im Original)
from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks

# REPRODUZIERBARKEIT
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# PARAMETER (Angepasst an Restormer Paper Konfiguration)
DEPTH = 5
SERIES_LEN = 41

# Paper Settings:
# Level 1 Kanäle: 48
# Expansion Factor GDFN: 2.66
EMBED_DIM = 48 
FFN_EXPANSION = 2.66
NUM_HEADS = [1, 2, 4, 8]  # Heads pro Level
NUM_BLOCKS = [4, 6, 6, 8] # Transformer Blöcke pro Level

BATCH_SIZE = 4
INITIAL_LR = 3e-4 # Paper startet mit 3e-4
EPOCHS = 100 # Anpassbar

FILES = {
    "training": "/home/sgaell/data/original_data/training_data.hdf5",
    "validation": "/home/sgaell/data/original_data/validation_data.hdf5",
}

TB_ROOT = Path.home() / "data" / "tblogs_restormer"
CKPT_FOLDER = "checkpoints_restormer"


# -----------------------------
# ARCHITEKTUR KOMPONENTEN (FUNKTIONAL)
# -----------------------------

def apply_bias_free_layernorm(x):
    """
    Paper Section 3.1: "We use bias-free convolutional layers... From a layer normalized tensor..."
    Implementation: Standard LayerNorm aber center=False (kein Bias/Beta).
    """
    return layers.LayerNormalization(epsilon=1e-6, center=False, scale=True)(x)

def mdta_block(x, dim, num_heads):
    input_tensor = x
    x = apply_bias_free_layernorm(x)
    
    # QKV Projektion
    qkv = layers.Conv2D(dim * 3, kernel_size=1, use_bias=False)(x)
    qkv = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(qkv)
    q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)
    
    # Statische Shapes extrahieren
    # Da dein Input-Shape fest ist (192, 240, 5), nutzen wir x.shape
    _, h, w, c = x.shape
    head_dim = dim // num_heads

    # Reshaping für Transposed Attention mit tf.reshape (funktioniert besser in Lambda)
    def transpose_attention(tensors):
        q_in, k_in, v_in = tensors
        b_dynamic = tf.shape(q_in)[0]
        
        # Flatten spatial: (B, HW, Heads, C_head)
        q_f = tf.reshape(q_in, (b_dynamic, -1, num_heads, head_dim))
        k_f = tf.reshape(k_in, (b_dynamic, -1, num_heads, head_dim))
        v_f = tf.reshape(v_in, (b_dynamic, -1, num_heads, head_dim))
        
        # Permute für Cross-Covariance: (B, Heads, C_head, HW)
        q_f = tf.transpose(q_f, (0, 2, 3, 1))
        k_f = tf.transpose(k_f, (0, 2, 3, 1))
        v_f = tf.transpose(v_f, (0, 2, 1, 3)) # (B, Heads, HW, C_head)
        
        # L2 Norm
        q_f = tf.math.l2_normalize(q_f, axis=-1)
        k_f = tf.math.l2_normalize(k_f, axis=-1)
        
        # Matmul: (B, Heads, C_head, C_head)
        attn = tf.matmul(q_f, k_f, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
        
        # Output: (B, Heads, HW, C_head)
        out = tf.matmul(v_f, attn, transpose_b=True)
        
        # Zurück zu (B, H, W, C)
        out = tf.transpose(out, (0, 2, 1, 3)) # (B, HW, Heads, C_head)
        return tf.reshape(out, (b_dynamic, h, w, dim))

    # Alles in einen Lambda-Layer packen, um Keras-Symbolik-Fehler zu vermeiden
    out = layers.Lambda(transpose_attention)([q, k, v])
    
    out = layers.Conv2D(dim, kernel_size=1, use_bias=False)(out)
    return layers.Add()([input_tensor, out])

def gdfn_block(x, dim, expansion_factor):
    """
    Gated-Dconv Feed-Forward Network (GDFN)
    Kontrolliert den Informationsfluss mittels Gating.
    """
    input_tensor = x
    hidden_dim = int(dim * expansion_factor)
    
    # 1. Norm
    x = apply_bias_free_layernorm(x)
    
    # 2. 1x1 Conv (Expansion)
    x = layers.Conv2D(hidden_dim * 2, kernel_size=1, use_bias=False)(x)
    
    # 3. 3x3 Depthwise Conv
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(x)
    
    # 4. Gating Mechanism
    # Split in zwei Pfade
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    
    # Gating: phi(x1) * x2, wobei phi = GELU
    x1 = layers.Activation('gelu')(x1)
    x = layers.Multiply()([x1, x2])
    
    # 5. 1x1 Conv (Projection back)
    x = layers.Conv2D(dim, kernel_size=1, use_bias=False)(x)
    
    # Residual Connection
    return layers.Add()([input_tensor, x])

def transformer_block(x, dim, num_heads, expansion_factor):
    """ Kombiniert MDTA und GDFN """
    x = mdta_block(x, dim, num_heads)
    x = gdfn_block(x, dim, expansion_factor)
    return x

def pixel_unshuffle(x, block_size=2):
    """ Downsampling via space_to_depth """
    return tf.nn.space_to_depth(x, block_size=block_size)

def pixel_shuffle(x, block_size=2):
    """ Upsampling via depth_to_space """
    return tf.nn.depth_to_space(x, block_size=block_size)

def downsample_block(x, dim):
    """ Paper: Pixel-unshuffle downsampling """
    # Da PixelUnshuffle Channel x4 nimmt, Restormer aber nur x2 will,
    # muss man aufpassen. Im offiziellen Code swz30: 
    # PixelUnshuffle macht H/2, W/2, 4C.
    # Dann 1x1 Conv um auf 2C zu reduzieren.
    # ABER Paper sagt: "Input image ... convolution ... low-level feature embeddings"
    # Hier: Space-to-Depth -> dann Channel Anpassung.
    x = layers.Lambda(pixel_unshuffle, arguments={'block_size': 2})(x)
    # PixelUnshuffle vervierfacht Kanäle. Wir wollen aber Verdopplung (dim -> 2*dim).
    # Aktuell hat x: 4*dim Kanäle. Ziel: 2*dim.
    x = layers.Conv2D(dim * 2, kernel_size=1, use_bias=False)(x)
    return x

def upsample_block(x, dim):
    """ Paper: Pixel-shuffle upsampling """
    # Input hat dim Kanäle. Ziel ist dim/2 Spatial x2.
    # Conv auf 2*Ziel-Dim (also dim) -> PixelShuffle -> dim/2
    
    # Schritt 1: Conv expansion auf 2 * output_dim (was hier 'dim' // 2 ist * 4 für shuffle? Nein.)
    # PixelShuffle(r=2) erwartet C*r^2 Kanäle um C auszugeben.
    # Wir wollen Output Channel = dim // 2. Also Input für Shuffle muss (dim//2)*4 = 2*dim sein.
    
    x = layers.Conv2D(dim * 2, kernel_size=1, use_bias=False)(x)
    x = layers.Lambda(pixel_shuffle, arguments={'block_size': 2})(x)
    return x


def build_restormer(input_shape=(192, 240, 5), out_channels=1):
    """
    Erstellt das Restormer Modell exakt nach Paper-Architektur.
    """
    inputs = layers.Input(shape=input_shape)
    
    # 1. Initial Convolution
    # "Restormer first applies a convolution to obtain low-level feature embeddings F0"
    x = layers.Conv2D(EMBED_DIM, kernel_size=3, padding='same', use_bias=False)(inputs)
    
    encoder_feats = []
    
    # ENCODER (4 Levels)
    # Level 1
    for _ in range(NUM_BLOCKS[0]):
        x = transformer_block(x, EMBED_DIM, NUM_HEADS[0], FFN_EXPANSION)
    encoder_feats.append(x)
    
    # Level 2 (Downsample -> Transformer)
    x = downsample_block(x, EMBED_DIM) # Dim: 48 -> 96
    for _ in range(NUM_BLOCKS[1]):
        x = transformer_block(x, EMBED_DIM * 2, NUM_HEADS[1], FFN_EXPANSION)
    encoder_feats.append(x)
    
    # Level 3
    x = downsample_block(x, EMBED_DIM * 2) # Dim: 96 -> 192
    for _ in range(NUM_BLOCKS[2]):
        x = transformer_block(x, EMBED_DIM * 4, NUM_HEADS[2], FFN_EXPANSION)
    encoder_feats.append(x)
    
    # Level 4 (Bottleneck)
    x = downsample_block(x, EMBED_DIM * 4) # Dim: 192 -> 384
    for _ in range(NUM_BLOCKS[3]):
        x = transformer_block(x, EMBED_DIM * 8, NUM_HEADS[3], FFN_EXPANSION)
    
    # DECODER (Symmetrisch)
    # Decoder Level 3 (Upsample -> Concat -> Reduce -> Transformer)
    x = upsample_block(x, EMBED_DIM * 8) # Output Channels: 192
    
    # Skip Connection
    skip = encoder_feats.pop() # Level 3 Skip
    x = layers.Concatenate()([x, skip])
    
    # Channel Reduction by half
    x = layers.Conv2D(EMBED_DIM * 4, kernel_size=1, use_bias=False)(x)
    
    for _ in range(NUM_BLOCKS[2]):
        x = transformer_block(x, EMBED_DIM * 4, NUM_HEADS[2], FFN_EXPANSION)
        
    # Decoder Level 2
    x = upsample_block(x, EMBED_DIM * 4) # Output Channels: 96
    skip = encoder_feats.pop()
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(EMBED_DIM * 2, kernel_size=1, use_bias=False)(x)
    
    for _ in range(NUM_BLOCKS[1]):
        x = transformer_block(x, EMBED_DIM * 2, NUM_HEADS[1], FFN_EXPANSION)
        
    # Decoder Level 1
    x = upsample_block(x, EMBED_DIM * 2) # Output Channels: 48
    skip = encoder_feats.pop()
    x = layers.Concatenate()([x, skip])
    
    # Paper Section 4.5 Table 8: "To aggregate encoder features with decoder at level-1, we do NOT employ 1x1 convolution"
    # Also KEINE Reduktion hier!
    
    for _ in range(NUM_BLOCKS[0]):
        x = transformer_block(x, EMBED_DIM * 2, NUM_HEADS[0], FFN_EXPANSION) # Channels sind hier 2xEMBED_DIM wegen Concat
        
    # REFINEMENT STAGE
    # "Refinement stage operating at high spatial resolution"
    for _ in range(4): # 4 Blocks im Refinement laut Paper Implementation Details
        x = transformer_block(x, EMBED_DIM * 2, NUM_HEADS[0], FFN_EXPANSION)
        
    # OUTPUT
    # "Finally, a convolution layer is applied... to generate residual image R"
    residual = layers.Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False)(x)
    
    # Add Degraded Image (Residual Learning)
    # Da Input 5 Layer (2.5D) ist, nehmen wir den mittleren Slice als Basis
    center_idx = input_shape[-1] // 2
    input_center = inputs[:, :, :, center_idx:center_idx+1]
    
    out = layers.Add()([input_center, residual])
    
    # Optional: Activation wenn du [0,1] erzwingen willst (Paper nutzt oft linearen Output für Residuals)
    out = layers.Activation('sigmoid')(out)
    
    return models.Model(inputs, out, name="Restormer_Exact_Functional")


# WARMUP & DATA LOADING
def lr_warmup_scheduler(epoch, lr):
    warmup_epochs = 10
    if epoch < warmup_epochs:
        return INITIAL_LR * (epoch + 1) / warmup_epochs
    return lr

def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:]
        high_count = f["high_count/data"][:]
    low_count = np.moveaxis(low_count, -1, 0)
    high_count = np.moveaxis(high_count, -1, 0)
    low_count = low_count[:, :, :, np.newaxis]
    high_count = high_count[:, :, :, np.newaxis]
    return low_count, high_count

def make_sliding_windows(X, y, series_len=None, depth=None):
    N, H, W, C = X.shape
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1
    X_v, y_v = [], []
    for i in range(n_series):
        start = i * series_len
        bX, bY = X[start:start+series_len], y[start:start+series_len]
        for start_idx in range(n_vols_per_series):
            X_v.append(bX[start_idx : start_idx + depth])
            y_v.append(bY[start_idx : start_idx + depth])
    return np.stack(X_v, axis=0), np.stack(y_v, axis=0)

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p=0.5):
    def map_volume(x, y):
        flip = tf.random.uniform([], 0.0, 1.0) < tf.constant(p, tf.float32)
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)
        x, y = tf.nn.relu(x), tf.nn.relu(y)
        sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        x, y = x / sum_x, y / sum_y
        scale = tf.random.uniform([], scale_min, scale_max)
        return x * scale, y * scale
    return map_volume

def prepare_restormer_input(x, y):
    x = tf.squeeze(x, axis=-1)
    x = tf.transpose(x, [1, 2, 0]) 
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

# Metriken
def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_val = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_val)

def mae_center(y_true, y_pred):
    y_true, y_pred = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mse_center(y_true, y_pred):
    y_true, y_pred = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def psnr_center(y_true, y_pred):
    y_true, y_pred = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(y_true, y_pred):
    y_true, y_pred = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))



# MAIN
print("Lade Daten...")
X_train, y_train = load_split(FILES["training"])
X_val, y_val = load_split(FILES["validation"])

X_train, y_train = make_sliding_windows(X_train, y_train, SERIES_LEN, DEPTH)
X_val, y_val = make_sliding_windows(X_val, y_val, SERIES_LEN, DEPTH)

X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
X_val, y_val = X_val.astype(np.float32), y_val.astype(np.float32)

BASE_NAME = "Restormer_Exact"
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{BASE_NAME}__BS{BATCH_SIZE}__emb{EMBED_DIM}__{RUN_ID}"

TB_RUN_DIR = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)
csv_path = Path.home() / "data" / CKPT_FOLDER / f"{RUN_NAME}.csv"

# Datasets
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
            .map(prepare_restormer_input, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
          .map(prepare_restormer_input, num_parallel_calls=tf.data.AUTOTUNE)
          .cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# Callbacks
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=2),
    make_epoch_ckpt_callback(RUN_NAME, folder_name=CKPT_FOLDER),
    tf.keras.callbacks.CSVLogger(str(csv_path), append=False),
    *tb_callbacks(TB_RUN_DIR),
]

# Build Model
model = build_restormer(input_shape=(192, 240, DEPTH), out_channels=1)

# Optimizer & Compile
# Paper nutzt AdamW. In Keras oft "Adam" mit weight decay ausreichend oder tfa.optimizers.AdamW
# Hier Standard Adam wie in deinem Code, aber mit Paper-LR
optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, amsgrad=True)
model.compile(optimizer=optimizer, loss=mae_ssim_2d, metrics=[mae_center, mse_center, psnr_center, ssim_center])

#model.summary()

print(f"Training beginnt: {RUN_NAME}")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

meta = make_meta_dict(
    script_name=RUN_NAME, batch_size=BATCH_SIZE, epochs=EPOCHS, 
    optimizer=optimizer, learning_rate=INITIAL_LR, input_shape=(192, 240, DEPTH),
    extra={"model": "Restormer_Exact_Paper_Impl"}
)

finalize_run(model, history, RUN_NAME, meta, folder_name=CKPT_FOLDER)
print("Training beendet.")