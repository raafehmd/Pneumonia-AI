from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam

class PneumoniaClassifier:
    def __init__(self, input_shape=(150, 150, 3)):
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_data, val_data, epochs):
        return self.model.fit(train_data, epochs=epochs, validation_data=val_data)

    def evaluate(self, test_data):
        return self.model.evaluate(test_data)

    def predict(self, test_data):
        return self.model.predict(test_data)