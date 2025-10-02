from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import LabelEncoder
from keras import models, layers

# 1. Load the TwoPatterns dataset
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('TwoPatterns')

# Check the new shape: (1000, 128, 1)
print(f"New X_train shape: {X_train.shape}")
print(f"New X_test shape: {X_test.shape}")

# 2. Encode the Labels (Still necessary as the labels are strings)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# 3. Adjust the CNN Model (Crucial changes for a Univariate dataset!)
# The new input shape is (128, 1)
cnn_new = models.Sequential([
    # Input shape is (Timesteps, Channels) -> (128, 1)
    layers.Conv1D(filters=32, kernel_size=8, activation='relu', input_shape=(128, 1)),
    layers.MaxPooling1D(pool_size=2), 

    layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),

    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    # TwoPatterns is a 4-class dataset
    layers.Dense(4, activation='softmax') # MUST be 4 classes for TwoPatterns
])

cnn_new.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

cnn_new.fit(X_train, y_train_encoded, epochs=10)
cnn_new.evaluate(X_test, y_test_encoded)