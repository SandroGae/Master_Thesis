# tensorboard.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os
import tensorflow as tf

def make_run_dir(
    run_name: str,
    *,
    root: Path | None = None,
    run_id: str | None = None,
    use_timestamp: bool = True,
) -> Path:
    """
    Legt einen eindeutigen Run-Ordner an:
    <root>/<run_name>__<run_id|timestamp>
    """
    root = Path(root) if root is not None else (Path.home() / "data" / f"tblogs_{run_name}")
    root.mkdir(parents=True, exist_ok=True)

    if run_id is None and use_timestamp:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if run_id is None:
        run_id = os.environ.get("RUN_ID", "run")

    run_dir = root / f"{run_name}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def tb_callbacks(
    run_dir: Path,
    *,
    histograms: bool = False,
    profile: bool = False,
    write_graph: bool = True,
    update_freq: str = "epoch",
) -> list[tf.keras.callbacks.Callback]:
    """
    Baut die Standard-Callbacks für TensorBoard (Scalars, Graph, optional Profiling/Histograms).
    """
    histogram_freq = 1 if histograms else 0
    profile_batch = (50, 60) if profile else 0
    return [
        tf.keras.callbacks.TensorBoard(
            log_dir=str(run_dir),
            histogram_freq=histogram_freq,
            write_graph=write_graph,
            update_freq=update_freq,
            profile_batch=profile_batch,
            write_images=False,  # Bilder separat loggen (siehe ImageLogger)
        )
    ]


class ImageLogger(tf.keras.callbacks.Callback):
    """
    Loggt Eingabe/Ziel/Vorhersage als Bilder (Images) in TensorBoard.
    Erwartet (x_vis,y_vis) als (N,H,W,1|3), float32 in [0,1].
    """
    def __init__(
        self,
        run_dir: Path,
        sample_batch,
        tag_prefix: str = "val",
        every_n_epochs: int = 1,
        max_outputs: int = 3,
    ):
        super().__init__()
        self.x, self.y = sample_batch
        self.every = max(1, int(every_n_epochs))
        self.max_outputs = int(max_outputs)
        self.writer = tf.summary.create_file_writer(str(Path(run_dir) / "images"))
        self.tag = tag_prefix
        # clamp/cast für TB
        self.x = tf.clip_by_value(tf.cast(self.x, tf.float32), 0.0, 1.0)
        self.y = tf.clip_by_value(tf.cast(self.y, tf.float32), 0.0, 1.0)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0:
            return
        # Vorhersage (prediction)
        p = self.model.predict(self.x, verbose=0)
        p = tf.clip_by_value(tf.cast(p, tf.float32), 0.0, 1.0)
        with self.writer.as_default():
            tf.summary.image(f"{self.tag}/x", self.x, step=epoch, max_outputs=self.max_outputs)
            tf.summary.image(f"{self.tag}/y", self.y, step=epoch, max_outputs=self.max_outputs)
            tf.summary.image(f"{self.tag}/p", p,   step=epoch, max_outputs=self.max_outputs)
