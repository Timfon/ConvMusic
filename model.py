from keras.layers import Input, BatchNormalization, Dropout, Dense, Flatten, Reshape  # type: ignore
from keras.models import Model, Sequential, load_model  # type: ignore
from preprocess import extract_decibels, preprocess_split, MAX_SONG_LENGTH, TIME_QUANTA
import numpy as np
import keras
import tensorflow as tf

OUTPUT_SHAPE = (None, int(MAX_SONG_LENGTH / TIME_QUANTA))

BEATMAPS_PATH = "beatmaps"
TIMESTAMP_MODEL_PATH = "timestamp.keras"
POSITION_MODEL_PATH = "position.keras"


@keras.saving.register_keras_serializable(package="convmusic")
class TimestampModel(Model):

    def __init__(self):
        super(TimestampModel, self).__init__()
        INPUT_SHAPE = (int(MAX_SONG_LENGTH / TIME_QUANTA), 1)
        DROPOUT_RATIO = 0.1

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

    def get_config(self):
        return {}


def train_timestamp_model(TRAIN_X: np.ndarray, TRAIN_Y: np.ndarray,
                          TEST_X: np.ndarray, TEST_Y: np.ndarray):
    model = TimestampModel()
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.summary()

    def process_targets(data_Y: np.ndarray) -> np.ndarray:
        return np.array([[(1 if y[2] > 0 else 0) for y in x] for x in data_Y])

    TRAIN_Y = process_targets(TRAIN_Y)
    TEST_Y = process_targets(TEST_Y)

    # Train the model
    model.fit(TRAIN_X,
              TRAIN_Y,
              epochs=10,
              batch_size=32,
              validation_data=(TEST_X, TEST_Y))

    # Save the model
    model.save(TIMESTAMP_MODEL_PATH)


@keras.saving.register_keras_serializable(package="convmusic")
class PositionModel(Model):

    def __init__(self):
        super(PositionModel, self).__init__()

        # Concatenate db list + encoded time
        INPUT_SHAPE = (2 * int(MAX_SONG_LENGTH / TIME_QUANTA), )

        HIDDEN_DIM = 128
        LATENT_DIM = int(MAX_SONG_LENGTH / TIME_QUANTA) // 4

        self.encoder = Sequential([
            Input(shape=INPUT_SHAPE),
            Dense(HIDDEN_DIM, activation='relu'),
            Dense(HIDDEN_DIM, activation='relu'),
            Dense(HIDDEN_DIM, activation='relu'),
        ])

        self.mu = Dense(LATENT_DIM, activation='linear')
        self.log_var = Dense(LATENT_DIM, activation='linear')

        self.decoder = Sequential([
            Dense(HIDDEN_DIM, activation='relu'),
            Dense(HIDDEN_DIM, activation='relu'),
            Dense(HIDDEN_DIM, activation='relu'),
            Dense(INPUT_SHAPE[0], activation='relu'),
            Reshape(INPUT_SHAPE)
        ])

    def call(self, inputs):  # type: ignore
        x = self.encoder(inputs)
        mu = self.mu(x)
        log_var = self.log_var(x)

        z = mu + tf.exp(log_var / 2) * tf.random.normal(tf.shape(mu))

        output = self.decoder(z)
        return output

    def get_config(self):
        return {}


def train_position_model(TRAIN_Y: np.ndarray, TEST_Y: np.ndarray):

    model = PositionModel()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    def prepare_input(data_Y: np.ndarray) -> np.ndarray:
        input = []
        for current_y in data_Y:
            sample = []
            for y in current_y:
                sample += [y[0], y[1]]
            input.append(sample)
        return np.array(input)

    X = prepare_input(np.concatenate([TRAIN_Y, TEST_Y], axis=0))

    model.fit(X, X, epochs=50)

    # Save the model
    model.save(POSITION_MODEL_PATH)

    predictions = model.predict(X)
    predictions = predictions[0]

    print(predictions[0:100])
    print(X[0][0:100])


if __name__ == "__main__":
    import sys
    import os
    from pydub import AudioSegment

    if "--train" in sys.argv:
        print("Preprocessing beatmaps...")
        TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = preprocess_split(BEATMAPS_PATH)

        # train_timestamp_model(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y)

        train_position_model(TRAIN_Y, TEST_Y)
        sys.exit(0)

    if len(sys.argv) != 2:
        print("Usage: python model.py <song_file>")
        sys.exit(1)

    song_file = sys.argv[1]
    assert os.path.isfile(song_file), "Song file not found."

    if (song_file[-3:] == "ogg"):
        sound = AudioSegment.from_ogg(song_file)
    elif (song_file[-3:] == "mp3"):
        sound = AudioSegment.from_mp3(song_file)
    else:
        raise Exception(
            f'Unknown file ending {song_file[-3:]} encountered in extract decibels'
        )

    assert (os.path.isfile("timestamp.keras")
            or not os.path.isfile("position.keras")
            ), "Models not found. Please run with --train to train models."

    timestamp_model: TimestampModel = load_model(
        "timestamp.keras")  # type: ignore
    position_model: PositionModel = load_model(
        "position.keras")  # type: ignore

    decibels = extract_decibels(sound)

    prediction = timestamp_model.predict(decibels.reshape(1, -1))
    timestamps = list(map(lambda x: min(1, max(0, round(x))), prediction[0]))

    # Predict positions
    position_model_input = []
    for i in range(len(timestamps)):
        position_model_input += [decibels[i], timestamps[i]]
    position_model_input = np.array(position_model_input)

    prediction = position_model.predict(position_model_input.reshape(1, -1))

    print(max(prediction[0]))
    positions = list(map(lambda x: max(0, round(x)), prediction[0]))

    count = 0
    i = 0
    while i < len(timestamps):
        if timestamps[i] == 1:
            count += 1
            print(
                f"{timestamps[i]}: {positions[i * 2]}, {positions[i * 2 + 1]}")
        i += 1
