# jens_stuff.py
import tensorflow as tf

# KEINE imports von sich selbst! (Das war der Fehler)

class SumScaleNormalizer:
    """
    Normalisiert Bilder basierend auf der Summe ihrer Pixelintensitäten.
    Rekonstruiert basierend auf der Logik in helper_functions.py (CustomDataGenerator).
    """
    def __init__(
        self, 
        scale_range=[5000, 15000], 
        pre_offset=0.0, 
        normalize_label=True, 
        axis=(1, 2, 3), 
        batch_mode=True, 
        clip_before=[0., float("inf")], 
        clip_after=[0., 1.]
    ):
        # Parameter speichern
        self.scale_min = float(scale_range[0])
        self.scale_max = float(scale_range[1])
        self.pre_offset = pre_offset
        self.normalize_label = normalize_label
        self.axis = axis
        self.batch_mode = batch_mode
        self.clip_before = clip_before
        self.clip_after = clip_after
        self.epsilon = 1e-12

    def _norm(self, img):
        """
        Interne Normalisierungsfunktion:
        1. Clip (min, max) -> Entspricht np.clip(img, a_min=0., a_max=None)
        2. Summe berechnen -> np.sum(img)
        3. Normalisieren -> img / sum * scale
        4. Clip Output -> np.clip(img, 0., 1.)
        """
        # 1. Clip Before (ReLU)
        if self.clip_before:
            # float("inf") funktioniert in TF clip_by_value nicht direkt als Obergrenze,
            # wir nutzen tf.maximum für die Untergrenze.
            if self.clip_before[0] is not None:
                 img = tf.maximum(img, self.clip_before[0])
            if self.clip_before[1] != float("inf"):
                 img = tf.minimum(img, self.clip_before[1])

        # 2. Summe berechnen
        # axis bestimmt, worüber summiert wird (H, W, C)
        sums = tf.reduce_sum(img, axis=self.axis, keepdims=True)
        sums = tf.maximum(sums, self.epsilon) # Division durch Null verhindern

        # 3. Scale bestimmen
        # Im Training war das zufällig: np.random.randint(low, high).
        # Für Inferenz nutzen wir einen festen Wert (hier scale_min), 
        # da du im Inferenz-Code [5000, 5001] übergibst.
        target_scale = self.scale_min 

        # 4. Normalisierung
        img_norm = (img / sums) * target_scale

        # 5. Clip After (meistens [0, 1])
        if self.clip_after:
            img_norm = tf.clip_by_value(img_norm, self.clip_after[0], self.clip_after[1])

        return img_norm

    def map(self, x, y=None):
        """
        Wendet die Normalisierung auf x (und optional y) an.
        Erwartet TensorFlow Tensoren.
        """
        x = tf.cast(x, tf.float32)
        x_n = self._norm(x)

        if y is not None:
            y = tf.cast(y, tf.float32)
            if self.normalize_label:
                y_n = self._norm(y)
            else:
                y_n = y
            return x_n, y_n
        
        return x_n

    def inverse_map(self, x, length=None):
        """
        Versucht die Normalisierung umzukehren (nur Skalierung).
        """
        return x / self.scale_min