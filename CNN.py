from keras import models, layers
from sklearn.model_selection import train_test_split

def CNN(X_tot, y_tot):
    # 1. Split the dataset into Train + Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_tot, y_tot,
        test_size=0.2,          # 20% test
        stratify=y_tot,         # keep 1/0 ratio the same
        shuffle=True,           # break your class-ordered rows
        random_state=42         # reproducible
    )

    # 2. Adjust the CNN Model (Crucial changes for a Univariate dataset!)
    # The new input shape is (400, 13)
    cnn_new = models.Sequential([
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

    cnn_new.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])

    print(f"X_train shape: {X_train.shape}; y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}; y_test shape: {y_test.shape}")
    
    cnn_new.fit(X_train, y_train, epochs=10)
    cnn_new.evaluate(X_test, y_test)