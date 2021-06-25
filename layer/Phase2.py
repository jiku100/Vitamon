from re import M
from tensorflow.keras import Model, Sequential
import tensorflow.keras.layers as layers


class Phase2(Model):
    def __init__(self) -> None:
        super(Phase2, self).__init__()
        self.model = Sequential(
            [
                layers.Conv2D(input_shape = (224,224,7), filters = 32, kernel_size = (3,3)),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(32, (3,3)),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(64, (3,3)),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.AveragePooling2D(),
                layers.AveragePooling2D(),
                layers.AveragePooling2D(),
                layers.AveragePooling2D(),
                layers.Dropout(0.5),
                layers.Flatten(),
            ]
        )
    
    def call(self, x):
        x = self.model(x)
        return x
    
    def summary(self):
        self.model.summary()