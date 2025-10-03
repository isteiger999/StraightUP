from keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # to find the right amount of epochs and not overfit

def CNN_model(X_train, X_test, y_train, y_test):

    # Adjust the CNN Model (Crucial changes for a Univariate dataset!)
    # The new input shape is (400, 13)
    cnn = models.Sequential([
        # Input shape is (Timesteps, Channels) -> (400, 13)
        layers.Conv1D(filters=32, kernel_size=14, activation='relu', input_shape=(400, 13)),
        layers.MaxPooling1D(pool_size=2), 

        layers.Conv1D(filters=64, kernel_size=10, activation='relu'),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=128, kernel_size=8, activation='relu'),
        layers.MaxPooling1D(pool_size=2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='softmax') 
    ])

    cnn.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])

    print(f"X_train shape: {X_train.shape}; y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}; y_test shape: {y_test.shape}")

    ## find optimal amount of epochs with 'patience'
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),  # patience 10: means even though validation error might not decrease anymore, we still go 10 epochs further to check if it really increases or just local minimum
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)               # multiplies learning rate by 0.5 if after 3 epochs validation error does not decrease 
    ]

    history = cnn.fit(
        X_train, y_train,
        validation_split=0.2,      # or validation_data=(X_val, y_val)
        epochs=200,                # goes maximally up to 200 epochs
        batch_size=32,
        shuffle=True,              # Keras shuffles each epoch for arrays by default
        callbacks=callbacks
    )
    print("Eval Score:")
    cnn.evaluate(X_test, y_test)