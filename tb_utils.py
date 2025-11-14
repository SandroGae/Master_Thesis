# tensorboard.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import tensorflow as tf


def make_run_dir(run_name: str) -> Path:
    """
    Legt einen Run-Ordner an unter:
        ~/data/tblogs/<run_name>__<timestamp>
    """
    root = Path.home() / "data" / "tblogs"
    root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_dir = root / f"{run_name}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir



def tb_callbacks(run_dir: Path):

    return [tf.keras.callbacks.TensorBoard(log_dir=str(run_dir), histogram_freq=0,
            write_graph=True, update_freq="epoch", write_images=False,)]


class ImageLogger(tf.keras.callbacks.Callback):
    """
    Loggt fuer 3D-Volumes die mittlere Slice als Bild in TensorBoard.
    Erwartet sample_batch als (X_vols, Y_vols) mit Shape (N, D, H, W, 1).
    Normalisierung analog zur Validation-Pipeline:
      - ReLU (Clip < 0 auf 0)
      - pro Slice Normierung auf Summe 1 ueber (H,W,C)
      - anschliessend Skalierung mit 'scale' (default 10000.0)
    """
    def __init__(
        self,
        run_dir: Path,
        sample_batch,
        tag_prefix: str = "val",
        every_n_epochs: int = 1,
        max_outputs: int = 3,
        scale: float = 10000.0,   # wie in val_ds: (10000,10001)
    ):
        super().__init__()
        x_vol, y_vol = sample_batch          # (N, D, H, W, 1)
        self.every = max(1, int(every_n_epochs))
        self.max_outputs = int(max_outputs)
        self.writer = tf.summary.create_file_writer(str(run_dir))
        self.tag = tag_prefix
        self.scale = float(scale)

        # in float32 casten
        x_vol = tf.cast(x_vol, tf.float32)
        y_vol = tf.cast(y_vol, tf.float32)

        # gleiche Normalisierung wie in augment_and_normalize_3d_per_slice
        self.x_infer = self._normalize_volume(x_vol)   # was dem Modell gefuettert wird
        self.y_infer = self._normalize_volume(y_vol)

        # fuer Visualisierung in [0,1] clippen (TB erwartet das)
        self.x_vol = tf.clip_by_value(self.x_infer, 0.0, 1.0)
        self.y_vol = tf.clip_by_value(self.y_infer, 0.0, 1.0)

    def _normalize_volume(self, vol):
        """
        vol: (N, D, H, W, C)
        Schritte:
          - ReLU
          - Summe pro Slice ueber (H,W,C)
          - durch Summe teilen
          - mit self.scale multiplizieren
        """
        vol = tf.nn.relu(vol)
        sum_vol = tf.reduce_sum(vol, axis=[1, 2, 3], keepdims=True) + 1e-12
        vol = vol / sum_vol
        vol = vol * self.scale
        return vol

    def _center_slice(self, vol):
        """
        vol: (N, D, H, W, C) -> mittlere Slice: (N, H, W, C)
        """
        D = tf.shape(vol)[1]
        idx = D // 2
        return vol[:, idx, :, :, :]  # (N, H, W, C)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0:
            return

        # Vorhersage auf den **gleich normalisierten** Volumes
        p_vol = self.model.predict(self.x_infer, verbose=0)
        p_vol = tf.cast(p_vol, tf.float32)
        p_vol = tf.clip_by_value(p_vol, 0.0, 1.0)

        # Mittlere Slice aus Input, Target, Prediction (alle bereits auf [0,1] gecappt)
        x_img = self._center_slice(self.x_vol)
        y_img = self._center_slice(self.y_vol)
        p_img = self._center_slice(p_vol)

        with self.writer.as_default():
            tf.summary.image(f"{self.tag}/x", x_img, step=epoch, max_outputs=self.max_outputs)
            tf.summary.image(f"{self.tag}/y", y_img, step=epoch, max_outputs=self.max_outputs)
            tf.summary.image(f"{self.tag}/p", p_img, step=epoch, max_outputs=self.max_outputs)

