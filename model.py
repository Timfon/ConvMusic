from keras.layers import Input, LSTM, Conv1D, BatchNormalization, Dropout, Dense, Flatten  # type: ignore
from keras.models import Model, Sequential  # type: ignore
from keras.utils import pad_sequences  # type: ignore
from preprocess import preprocess_split, MAX_SONG_LENGTH, TIME_QUANTA
import tensorflow as tf  # type: ignore

INPUT_SHAPE = (int(MAX_SONG_LENGTH / TIME_QUANTA), 1)
OUTPUT_SHAPE = (None, int(MAX_SONG_LENGTH / TIME_QUANTA))
DROPOUT_RATIO = 0.1


class TimestampModel(Model):

    def __init__(self):
        super(TimestampModel, self).__init__()

        self.model = Sequential([
            Input(shape=INPUT_SHAPE),
            Flatten(),
            Dropout(DROPOUT_RATIO),
            BatchNormalization(),
            Dense(128, activation='leaky_relu'),
            Dense(OUTPUT_SHAPE[1], activation='leaky_relu')
        ])

    def call(self, inputs):  # type: ignore
        return self.model(inputs)


model = TimestampModel()
model.compile(optimizer='adam', loss='mean_absolute_error')
model.summary()

# Preprocess the data
TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = preprocess_split("beatmaps")
# TRAIN_X is a (split*len(sample), MAX_SONG_LENGTH/TIME_QUANTA) array

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
