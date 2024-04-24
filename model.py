from keras.layers import Input, BatchNormalization, Dropout, Dense, Flatten  # type: ignore
from keras.models import Model, Sequential, load_model  # type: ignore
from preprocess import extract_decibels, preprocess_split, MAX_SONG_LENGTH, TIME_QUANTA
from output import output_osu_string, output_osu_file
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

        self.model = Sequential([
            Input(shape=INPUT_SHAPE),
            Dense(128, activation='linear'),
            Dense(2 * OUTPUT_SHAPE[1], activation='linear')
        ])

    def call(self, inputs):  # type: ignore
        return self.model(inputs)

    def get_config(self):
        return {}

@keras.saving.register_keras_serializable()
def custom_mse(y_true, y_pred):
    print("True: ", y_true)
    print("True shape: ", y_true.shape)
    empty_entries = y_true == 0
    empty_entries = tf.cast(tf.where(empty_entries, 0, 1), tf.float32)
    y_pred *= empty_entries
    return keras.losses.mean_squared_error(y_true, y_pred)

def train_position_model(TRAIN_X: np.ndarray, TRAIN_Y: np.ndarray,
                         TEST_X: np.ndarray, TEST_Y: np.ndarray):
    model = PositionModel()
    model.compile(optimizer='adam', loss=custom_mse)
    model.summary()

    def prepare_features(data_X: np.ndarray, data_Y: np.ndarray) -> np.ndarray:
        new_data_X = []
        for current_x, current_y in zip(data_X, data_Y):
            sample = []
            for x, y in zip(current_x, current_y):
                sample += [x, 1 if y[2] > 0 else 0]
            new_data_X.append(sample)
        return np.array(new_data_X)

    def prepare_targets(data_Y: np.ndarray) -> np.ndarray:
        new_data_Y = []
        for current_y in data_Y:
            sample = []
            for y in current_y:
                sample += [y[0], y[1]]
            new_data_Y.append(sample)
        return np.array(new_data_Y)

    TRAIN_X = prepare_features(TRAIN_X, TRAIN_Y)
    TEST_X = prepare_features(TEST_X, TEST_Y)

    TRAIN_Y = prepare_targets(TRAIN_Y)
    TEST_Y = prepare_targets(TEST_Y)

    # Train the model
    model.fit(TRAIN_X,
              TRAIN_Y,
              epochs=20,
              batch_size=32,
              validation_data=(TEST_X, TEST_Y))

    # Save the model
    model.save(POSITION_MODEL_PATH)


if __name__ == "__main__":
    import sys
    import os
    from pydub import AudioSegment

    if "--train" in sys.argv:
        print("Preprocessing beatmaps...")
        TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = preprocess_split(BEATMAPS_PATH)

        train_timestamp_model(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y)

        train_position_model(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y)
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

    osu_string = output_osu_string(timestamps, positions, song_file)
    output_osu_file(osu_string, "output", song_file)
