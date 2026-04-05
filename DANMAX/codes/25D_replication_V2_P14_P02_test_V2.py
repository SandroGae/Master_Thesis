#!/usr/bin/env python3

import os
import random
import gc
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from unet_3d_simple_checkpoints import finalize_run, make_meta_dict
from tb_utils import tb_callbacks

# =====================================================
# PARAMETER CONFIGURATION
# =====================================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

DATA_SPLIT_SEED = 42 

DEPTH = 5
SERIES_LEN = 40 
BASEFILTERS = 64
CROP_SIZE = (512, 512)

EPOCHS = 200
LR_TARGET = 5e-4
WARMUP_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 25
RLROP_PATIENCE = 15
BATCH_SIZE = 8

# P14 und P02 Konfiguration (Alpha/Beta Vertauschung)
CONFIGS = [
    {"point": "P14", "alpha": 2.0/6.0, "beta": 0.0},
    {"point": "P02", "alpha": 0.0, "beta": 2.0/6.0}
]

# =====================================================
# ARCHITEKTUR
# =====================================================
def conv_block_2d(x, filters, kernel_size=(3, 3), padding="same"):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(512, 512, DEPTH), base_filters=BASEFILTERS, output_activation="sigmoid"):
    inputs = layers.Input(shape=input_shape, name="input")

    c1 = conv_block_2d(inputs, base_filters)          ; p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, base_filters * 2)          ; p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, base_filters * 4)          ; p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, base_filters * 8)          ; p4 = layers.MaxPooling2D((2, 2))(c4)

    bn = conv_block_2d(p4, base_filters * 16)

    u4 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])               ; c5 = conv_block_2d(u4, base_filters * 8)
    u3 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])               ; c6 = conv_block_2d(u3, base_filters * 4)
    u2 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])               ; c7 = conv_block_2d(u2, base_filters * 2)
    u1 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])               ; c8 = conv_block_2d(u1, base_filters)

    out = layers.Conv2D(1, (1, 1), activation=output_activation, name="output")(c8) 
    
    return models.Model(inputs, out, name="unet_25d_stacked")

# =====================================================
# HILFSFUNKTIONEN DATEN
# =====================================================
def load_and_correct_danmax(base_path, scan_id, crop_size=CROP_SIZE):
    ct_file    = os.path.join(base_path, f"scan-{scan_id:04d}_orca.h5")
    white_file = os.path.join(base_path, f"scan-{scan_id-1:04d}_orca.h5")
    dark_file  = os.path.join(base_path, f"scan-{scan_id-2:04d}_orca.h5")
    data_path  = 'entry/instrument/orca/data'
    
    with h5py.File(dark_file, 'r') as f_d, h5py.File(white_file, 'r') as f_w, h5py.File(ct_file, 'r') as f_ct:
        m_dark  = np.mean(f_d[data_path][:], axis=0).astype(np.float32)
        m_white = np.mean(f_w[data_path][:], axis=0).astype(np.float32)
        
        full_h, full_w = f_ct[data_path].shape[1], f_ct[data_path].shape[2]
        h_s = (full_h - crop_size[0]) // 2
        w_s = (full_w - crop_size[1]) // 2
        
        projs = f_ct[data_path][:2000, h_s:h_s+crop_size[0], w_s:w_s+crop_size[1]].astype(np.float32)
        m_dark_c  = m_dark[h_s:h_s+crop_size[0], w_s:w_s+crop_size[1]]
        m_white_c = m_white[h_s:h_s+crop_size[0], w_s:w_s+crop_size[1]]
        
        denom = m_white_c - m_dark_c
        denom[denom < 1e-6] = 1e-6
        corrected = (projs - m_dark_c) / denom
        return np.clip(corrected, 0, 1)

def load_split_danmax(base_path, gt_id, low_id):
    high_count = load_and_correct_danmax(base_path, gt_id)
    low_count  = load_and_correct_danmax(base_path, low_id)
    return low_count[..., np.newaxis], high_count[..., np.newaxis]

# --- INTELLIGENTE GENERATOR LOGIK (NEU) ---
def get_valid_indices(num_slices, series_len, depth):
    indices = []
    n_series = num_slices // series_len
    for i in range(n_series):
        start_idx = i * series_len
        n_vols = series_len - depth + 1
        for s_idx in range(n_vols):
            indices.append(start_idx + s_idx)
    return np.array(indices)

def get_dynamic_generator(X_raw, y_raw, indices, depth=DEPTH, shuffle_every_epoch=False):
    def gen():
        current_indices = indices.copy()
        if shuffle_every_epoch:
            np.random.shuffle(current_indices)
        
        for idx in current_indices:
            yield X_raw[idx : idx + depth], y_raw[idx : idx + depth]
    return gen

# --- ALTE NORMALISIERUNG (BEIBEHALTEN WIE GEWÜNSCHT) ---
def augment_and_normalize_3d_per_slice(p: float, phys_max: float):
    def map_volume(x, y):
        flip = tf.random.uniform([], 0, 1) < p
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)
        
        x = tf.nn.relu(x); y = tf.nn.relu(y)
        
        x = x / (tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12)
        y = y / (tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12)
        
        x = x / phys_max
        y = y / phys_max
        
        x = tf.clip_by_value(x, 0.0, 1.0)
        y = tf.clip_by_value(y, 0.0, 1.0)
        
        return x, y
    return map_volume

def prepare_25d_input(x, y):
    x = tf.transpose(tf.squeeze(x, axis=-1), [1, 2, 0])
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

# =====================================================
# METRIKEN & LOSS
# =====================================================
def get_triple_loss(alpha, beta):
    def loss(yt, yp):
        mae = tf.reduce_mean(tf.abs(yt - yp))
        mse = tf.reduce_mean(tf.square(yt - yp))
        ssim = 1.0 - tf.reduce_mean(tf.image.ssim(yt, yp, 1.0))
        return (alpha * ssim) + ((1.0 - alpha) * (beta * mse + (1.0 - beta) * mae))
    return loss

def mae_clipped(y_true, y_pred): return tf.reduce_mean(tf.abs(y_true - y_pred))
def mse_clipped(y_true, y_pred): return tf.reduce_mean(tf.math.square(y_true - y_pred))
def psnr_clipped(y_true, y_pred):
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)
def ssim_clipped(y_true, y_pred): return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def lr_warmup_scheduler(epoch, lr):
    return LR_TARGET * (epoch + 1) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else lr

# =====================================================
# HAUPTSCHLEIFE
# =====================================================
def main():
    print("Lade Daten von DanMAX (Bamboo)...")
    BAMBOO_RAW = Path("/scratch/sgaell/DATA_DANMAX/2026020508/raw/bamboo")

    X_all, y_all = load_split_danmax(BAMBOO_RAW, gt_id=32, low_id=57)

    print("Erstelle reproduzierbaren 60/20/20 Split auf Basis von 40er Serien...")
    N_SERIES = len(X_all) // SERIES_LEN

    X_series = np.reshape(X_all, (N_SERIES, SERIES_LEN, CROP_SIZE[0], CROP_SIZE[1], 1))
    y_series = np.reshape(y_all, (N_SERIES, SERIES_LEN, CROP_SIZE[0], CROP_SIZE[1], 1))

    rng = np.random.default_rng(DATA_SPLIT_SEED)
    indices = np.arange(N_SERIES)
    rng.shuffle(indices)

    X_series = X_series[indices]
    y_series = y_series[indices]

    n_train = int(0.6 * N_SERIES)
    n_val = int(0.2 * N_SERIES)

    # Raw-Daten behalten (4D-Arrays), spart enorm RAM
    X_train_raw = np.reshape(X_series[:n_train], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
    y_train_raw = np.reshape(y_series[:n_train], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))

    X_val_raw = np.reshape(X_series[n_train:n_train+n_val], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
    y_val_raw = np.reshape(y_series[n_train:n_train+n_val], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))

    X_test_raw = np.reshape(X_series[n_train+n_val:], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
    y_test_raw = np.reshape(y_series[n_train+n_val:], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))

    # =====================================================
    # DYNAMISCHE BERECHNUNG DES PEAKS (Angepasst für 4D Raw-Arrays)
    # =====================================================
    print("\nBerechne optimalen Skalierungsfaktor (Robust gegen Outlier)...")
    def get_peak(data):
        # Data shape hier ist (N, 512, 512, 1)
        sums = np.sum(data, axis=(1, 2, 3), keepdims=True) + 1e-12
        return np.percentile(data / sums, 99.99)

    peak_y = get_peak(y_train_raw)
    peak_x = get_peak(X_train_raw)

    GLOBAL_PEAK_DIVISOR = max(peak_y, peak_x)
    PHYSICAL_MAX = float(GLOBAL_PEAK_DIVISOR * 1.02)

    print(f"-> Robustes Peak-Niveau (99.99% Quantil): {GLOBAL_PEAK_DIVISOR:.6f}")
    print(f"-> PHYSICAL_MAX wird: {PHYSICAL_MAX:.6f}\n")

    # =====================================================
    # GENERATOREN BEREITSTELLEN
    # =====================================================
    print("Berechne Start-Indizes für den dynamischen Generator...")
    train_indices = get_valid_indices(len(X_train_raw), SERIES_LEN, DEPTH)
    val_indices   = get_valid_indices(len(X_val_raw), SERIES_LEN, DEPTH)
    test_indices  = get_valid_indices(len(X_test_raw), SERIES_LEN, DEPTH)

    # Initiales Mischen der Train-Indizes
    rng = np.random.default_rng(SEED)
    rng.shuffle(train_indices)

    output_sig = (
        tf.TensorSpec(shape=(DEPTH, CROP_SIZE[0], CROP_SIZE[1], 1), dtype=tf.float32),
        tf.TensorSpec(shape=(DEPTH, CROP_SIZE[0], CROP_SIZE[1], 1), dtype=tf.float32)
    )

    train_ds = (tf.data.Dataset.from_generator(
                    get_dynamic_generator(X_train_raw, y_train_raw, train_indices, shuffle_every_epoch=True), 
                    output_signature=output_sig)
                .map(augment_and_normalize_3d_per_slice(p=0.5, phys_max=PHYSICAL_MAX), num_parallel_calls=tf.data.AUTOTUNE)
                .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_generator(
                    get_dynamic_generator(X_val_raw, y_val_raw, val_indices), 
                    output_signature=output_sig)
              .map(augment_and_normalize_3d_per_slice(p=0.0, phys_max=PHYSICAL_MAX), num_parallel_calls=tf.data.AUTOTUNE)
              .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    test_ds = (tf.data.Dataset.from_generator(
                    get_dynamic_generator(X_test_raw, y_test_raw, test_indices), 
                    output_signature=output_sig)
               .map(augment_and_normalize_3d_per_slice(p=0.0, phys_max=PHYSICAL_MAX), num_parallel_calls=tf.data.AUTOTUNE)
               .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    # =====================================================
    # TRAINING SCHLEIFE (P14 & P02)
    # =====================================================
    RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    for config in CONFIGS:
        point_name = config["point"]
        alpha = config["alpha"]
        beta = config["beta"]
        
        RUN_NAME = f"unet_25d_OLDLOGIC_RAM_FIX_{point_name}__seed{SEED}__alpha{alpha:.2f}_beta{beta:.2f}__{RUN_ID}"
        print(f"\n{'='*60}")
        print(f"🚀 STARTE TRAINING FÜR {point_name} (Alpha: {alpha:.2f}, Beta: {beta:.2f})")
        print(f"{'='*60}\n")

        TB_ROOT    = Path.home() / "scratch" / "DANMAX" / "codes" / "tb_root"
        TB_RUN_DIR = TB_ROOT / RUN_NAME
        TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

        MODEL_OUT_DIR = Path.home() / "scratch" / "DANMAX" / "codes" / "models"
        MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

        best_keras_file = MODEL_OUT_DIR / f"{RUN_NAME}_best_model.keras"
        best_weights_file = MODEL_OUT_DIR / f"{RUN_NAME}_best_weights.h5"

        # --- INTELLIGENTES & ROBUSTES SICHERHEITSNETZ ---
        status = {"best_psnr": -1.0, "drop_cnt": 0, "aborted": False, "reason": "none"}
        def check_crash(epoch, logs):
            loss = logs.get("loss", 0.0)
            val_loss = logs.get("val_loss", 0.0)
            psnr = logs.get("val_psnr_clipped", 0.0)

            # 1. Absoluter Not-Halt: NaN-Werte
            if np.isnan(loss) or np.isnan(val_loss) or np.isnan(psnr):
                status["aborted"] = True
                status["reason"] = "nan_detected"
                model.stop_training = True
                print(f"\n🚨 SICHERHEITSABBRUCH in Epoche {epoch+1}! Das Netzwerk spuckt NaN-Werte aus (Loss: {loss}, Val-Loss: {val_loss}).")
                return

            # 2. Toleranter PSNR-Kollaps-Check (erst wenn das Netz warmgelaufen ist)
            if psnr > status["best_psnr"]:
                status["best_psnr"] = psnr
                status["drop_cnt"] = 0  # Zähler sofort resetten, wenn es ein neues High gibt
            elif epoch >= 15:  # Geben wir ihm 15 Epochen Zeit zum Finden der Richtung
                # Wir greifen nur ein, wenn der PSNR um unfassbare 12 dB einbricht 
                # ODER auf unter 10.0 dB fällt (was faktisch schlimmer als reines Rauschen ist)
                if psnr < (status["best_psnr"] - 12.0) or psnr < 10.0:
                    status["drop_cnt"] += 1
                else:
                    status["drop_cnt"] = 0  # Wenn es sich auch nur leicht erholt, Zähler zurücksetzen!

                if status["drop_cnt"] >= 4:  # Es muss 4 Epochen am Stück komplett tot sein
                    status["aborted"] = True
                    status["reason"] = "massive_psnr_collapse"
                    model.stop_training = True
                    print(f"\n⚠️ SICHERHEITSABBRUCH in Epoche {epoch+1}! PSNR ist massiv auf {psnr:.2f} dB kollabiert (Bestwert: {status['best_psnr']:.2f} dB).")

        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
            tf.keras.callbacks.ModelCheckpoint(filepath=str(best_keras_file), monitor="val_loss", save_best_only=True, save_weights_only=False, mode="min", verbose=1),
            tf.keras.callbacks.ModelCheckpoint(filepath=str(best_weights_file), monitor="val_loss", save_best_only=True, save_weights_only=True, mode="min", verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=RLROP_PATIENCE, min_lr=1e-6, verbose=2),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=check_crash),
            tf.keras.callbacks.CSVLogger(str(TB_RUN_DIR / f"{RUN_NAME}.csv"), append=False),
            *tb_callbacks(TB_RUN_DIR),
        ]

        model = unet_2d_stacked(input_shape=(CROP_SIZE[0], CROP_SIZE[1], DEPTH)) 
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR_TARGET, amsgrad=True)

        model.compile(
            optimizer=optimizer, 
            loss=get_triple_loss(alpha, beta), 
            metrics=[mae_clipped, mse_clipped, psnr_clipped, ssim_clipped]
        )

        print(f"Training für {point_name} beginnt...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

        if status["aborted"]:
            print(f"❌ Training von {point_name} wurde vorzeitig beendet wegen: {status['reason']}")

        print(f"Evaluation auf dem Test-Set für {point_name}...")
        test_results = model.evaluate(test_ds, verbose=1, return_dict=True)

        meta = make_meta_dict(
            script_name=RUN_NAME, 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            optimizer=optimizer,
            learning_rate=LR_TARGET, 
            input_shape=(CROP_SIZE[0], CROP_SIZE[1], DEPTH),
            extra={
                "point": point_name,
                "alpha": alpha,
                "beta": beta,
                "warmup_epochs": WARMUP_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "rlrop_patience": RLROP_PATIENCE,
                "data_split_seed": DATA_SPLIT_SEED,
                "aborted": status["aborted"],
                "test_loss": float(test_results.get("loss", -1)),
                "test_psnr": float(test_results.get("psnr_clipped", -1))
            }
        )

        finalize_run(model, history, RUN_NAME, meta)
        
        # GPU RAM leeren, bevor das nächste Modell startet
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == "__main__":
    main()