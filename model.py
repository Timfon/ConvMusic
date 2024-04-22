from tensorflow.keras.layers import Input, LSTM, Conv1D, BatchNormalization, Dropout, Dense  # type: ignore
from tensorflow.keras.models import Model, Sequential  # type: ignore
from tensorflow.keras.utils import pad_sequences  # type: ignore
from preprocess import preprocess_split, N_FFT
import tensorflow as tf  # type: ignore

INPUT_SHAPE = (None, int((N_FFT / 2) + 1))
OUTPUT_SHAPE = (None, 1)
DROPOUT_RATIO = 0.1


class BeatmapModel(Model):

    def __init__(self):
        super(BeatmapModel, self).__init__()

        self.model = Sequential([
            Input(shape=INPUT_SHAPE),
            Conv1D(64,
                   kernel_size=32,
                   activation='relu',
                   padding='same',
                   strides=2),
            # Dropout(DROPOUT_RATIO),
            BatchNormalization(),
            Dense(128, activation='leaky_relu'),
            Dense(OUTPUT_SHAPE[1], activation='leaky_relu')
        ])

    def call(self, inputs):
        return self.model(inputs)


model = BeatmapModel()
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Preprocess the data
TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = preprocess_split("beatmaps")

# Train the model
model.fit(TRAIN_X,
          TRAIN_Y,
          epochs=10,
          batch_size=32,
          validation_data=(TEST_X, TEST_Y))

# Save the model
model.save("model.keras")

# Run the model
predictions = model.predict(TEST_X)
print(TEST_Y[0][0:48])
print(predictions[0][0:48])
