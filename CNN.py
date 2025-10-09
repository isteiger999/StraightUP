import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import coremltools as ct
import numpy as np

def CNN_model(X_train, y_train):

    # Regularization for weights inside filters
    l2 = regularizers.l2(1e-4)

    # Adding Normalization layer into CNN and train it on RAW X_train (yes X_train itself is not normalized)
    norm = layers.Normalization(axis=-1)
    norm.adapt(X_train.astype("float32"))  # TRAIN ONLY
    
    # The new input shape is (400, 14)
    cnn = models.Sequential([
        layers.Input(shape=(75, 13)),
        norm,
        layers.Conv1D(filters=32, kernel_size=11, padding="same", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.Conv1D(64, 9, padding="same", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.Conv1D(96, 7, padding="same", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.30),                   # means random 30% of neurons get deactivated
        layers.Dense(64, activation="relu", kernel_regularizer=l2),
        layers.Dropout(0.20),
        layers.Dense(1, activation="sigmoid")   # predicts P(class=1) (thats why output only 1D)
    ])

    cnn.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                tf.keras.metrics.AUC(name="roc_auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc")]
    )

    ## find optimal amount of epochs with 'patience'          
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                        patience=10, restore_best_weights=True), # patience 10: means even though validation error might not decrease anymore, we still go 10 epochs further to check if it really increases or just local minimum
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max",
                                            factor=0.5, patience=3)              # multiplies learning rate by 0.5 if after 3 epochs validation error does not decrease 
    ]

    cnn.fit(
        X_train, y_train,
        validation_split=0.2,      # or validation_data=(X_val, y_val)
        epochs=200,                # goes maximally up to 200 epochs
        batch_size=32,
        shuffle=True,              # Keras shuffles each epoch for arrays by default
        callbacks=callbacks
    )
    
    return cnn


def export_coreml(model, out_path="PostureCNN.mlpackage"):
    mlmodel = ct.convert(
        model,
        source="tensorflow",
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=model.inputs[0].name.split(":")[0],
                              shape=(1, 400, 14), dtype=np.float32)]
    )
    mlmodel.save(out_path)   # <- .mlpackage
    print(f"✅ Saved {out_path}")