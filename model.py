from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, LSTM, MaxPooling2D, Reshape, TimeDistributed  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from preprocess import preprocess_split
import numpy as np

SPECTROGRAM_SHAPE = (1025, None)
HIT_OBJECTS_SHAPE = (None, 5)


class BeatmapModel(Model):

    def __init__(self, output_sequence_length=100, output_feature_dim=5):
        super(BeatmapModel, self).__init__()

        self.spectrogram_input = Input(shape=SPECTROGRAM_SHAPE)

        self.conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling2D((2, 2), padding='same')
        self.conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = MaxPooling2D((2, 2), padding='same')
        self.flatten = Flatten()

        self.reshape = Reshape((output_sequence_length, -1))

        self.lstm1 = LSTM(128, return_sequences=True)
        self.lstm2 = LSTM(64, return_sequences=True)

        self.time_dist = TimeDistributed(Dense(output_feature_dim))

    def call(self, inputs):
        # Feature extraction
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)

        # Reshape for sequence generation
        x = self.reshape(x)

        # Sequence generation
        x = self.lstm1(x)
        x = self.lstm2(x)

        # TimeDistributed layer for output
        return self.time_dist(x)


model = BeatmapModel()
model.compile(optimizer='adam', loss='mean_squared_error')

# Preprocess the data
TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = preprocess_split("beatmaps")

padded = pad_sequences(TRAIN_X, padding='post', dtype='float32')
TRAIN_X = np.expand_dims(padded, axis=0)
print(TRAIN_X.shape)

padded = pad_sequences(TRAIN_Y, padding='post', dtype='float32')
TRAIN_Y = np.expand_dims(padded, axis=0)
print(TRAIN_Y.shape)

# Train the model
model.fit(TRAIN_X,
          TRAIN_Y,
          epochs=10,
          batch_size=32,
          validation_data=(TEST_X, TEST_Y))

# Run the model
predictions = model.predict(TEST_X)
print(predictions)
