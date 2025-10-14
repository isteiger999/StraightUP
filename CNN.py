from constants import set_seeds, configure_tensorflow
set_seeds()
configure_tensorflow()
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight
import coremltools as ct
import numpy as np

def balanced_accuracy(y_true, y_pred, n_classes=3):
    y_true = tf.cast(tf.squeeze(y_true), tf.int32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=n_classes, dtype=tf.float32)
    per_class_recall = tf.linalg.diag_part(cm) / (tf.reduce_sum(cm, axis=1) + 1e-9)
    return tf.reduce_mean(per_class_recall)

class BalancedAccuracy(tf.keras.metrics.Metric):
    '''Balanced Accuracy is a classification metric that measures a model's 
    performance by giving equal importance to every class, regardless of how 
    many samples each class has in the dataset.
    Imagine a dataset of 100 images: 90 of Class A and 10 of Class B.
    90% accuary is misleading. For 3 label problem as ours, we need balanced accuracy > 0.33,
    otherwise random guessing would be even better'''
    def __init__(self, n_classes=3, name="balanced_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        # Per-class true positives and total positives
        self.tp = self.add_weight(name="tp", shape=(n_classes,), initializer="zeros")
        self.pos = self.add_weight(name="pos", shape=(n_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        y_true_oh = tf.one_hot(y_true, depth=self.n_classes, dtype=self.dtype)
        y_pred_oh = tf.one_hot(y_pred, depth=self.n_classes, dtype=self.dtype)

        tp_batch  = tf.reduce_sum(y_true_oh * y_pred_oh, axis=0)   # diag of CM
        pos_batch = tf.reduce_sum(y_true_oh, axis=0)               # row sums of CM

        if sample_weight is not None:
            sw = tf.cast(tf.reshape(sample_weight, (-1, 1)), self.dtype)
            tp_batch  = tf.reduce_sum(sw * (y_true_oh * y_pred_oh), axis=0)
            pos_batch = tf.reduce_sum(sw * y_true_oh, axis=0)

        self.tp.assign_add(tp_batch)
        self.pos.assign_add(pos_batch)

    def result(self):
        per_class_recall = self.tp / (self.pos + tf.keras.backend.epsilon())
        return tf.reduce_mean(per_class_recall)

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))


def CNN_model(X_train, y_train, X_val, y_val, verbose, n_classes=3):
    T = X_train.shape[1]
    n_ch = X_train.shape[2]
    assert X_train.ndim == 3 and X_train.shape[1:] == (75, n_ch)
    assert X_val.shape[1:]   == (75, n_ch)

    # labels should be integer class IDs: 0,1,2
    y_train = y_train.squeeze().astype("int32")
    y_val   = y_val.squeeze().astype("int32")

    l2 = regularizers.l2(8e-4)

    # Per-feature normalization (fit on train only)
    norm = layers.Normalization(axis=-1)
    norm.adapt(X_train.astype("float32"))
    
    # used to be: 32, 64, 96
    cnn = models.Sequential([
        layers.Input(shape=(T, n_ch)),
        norm,
        layers.Conv1D(24, 9, padding="causal", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.Conv1D(48, 7,  padding="causal", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.Conv1D(96, 5,  padding="causal", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),
        
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.30),
        layers.Dense(64, activation="relu", kernel_regularizer=l2),
        layers.Dropout(0.20),
        layers.Dense(n_classes, activation="softmax")   # 3 logits -> probs
    ])
    ## These metrices are then shown in the cnn.eval on X_val and y_val
    cnn.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.legacy.Adam(5e-4), 
        metrics=[BalancedAccuracy(n_classes=3)]             # removed "accuracy" (14. October 2025)
    )
    
    monitor = "val_balanced_acc"  # ✅ matches the metric name above
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor=monitor, mode="max",
                                        patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode="max",
                                            factor=0.5, patience=5),
    ]

    # For class imbalances (we have much more 0 than 1 and 2)
    present = np.unique(y_train.ravel())
    valid = np.array([c for c in [0,1,2] if c in present])
    class_weight = None
    if valid.size:
        cw = compute_class_weight('balanced', classes=valid, y=y_train.ravel())
        class_weight = {int(c): float(w) for c, w in zip(valid, cw)}
    
    shuffle_idx = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[shuffle_idx]
    y_train_shuffled = y_train[shuffle_idx]

    cnn.fit(
        X_train_shuffled, y_train_shuffled,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
        class_weight=class_weight
    )

    return cnn


def export_coreml(X_train, model, out_path="PostureCNN.mlpackage"):
    T = X_train.shape[1]
    n_ch = X_train.shape[2]
    mlmodel = ct.convert(
        model,
        source="tensorflow",
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=model.inputs[0].name.split(":")[0],
                              shape=(1, T, n_ch), dtype=np.float32)]
    )
    mlmodel.save(out_path)   # <- .mlpackage
    print(f"✅ Saved {out_path}")