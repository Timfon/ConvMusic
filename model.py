from keras.layers import Input, BatchNormalization, Dropout, Dense, Flatten  # type: ignore
from keras.models import Model, Sequential, load_model  # type: ignore
from preprocess import extract_decibels, preprocess_split, MAX_SONG_LENGTH, TIME_QUANTA
import numpy as np
import keras

OUTPUT_SHAPE = (None, int(MAX_SONG_LENGTH / TIME_QUANTA))
DROPOUT_RATIO = 0.1


@keras.saving.register_keras_serializable(package="convmusic")
class TimestampModel(Model):

    def __init__(self):
        super(TimestampModel, self).__init__()
        INPUT_SHAPE = (int(MAX_SONG_LENGTH / TIME_QUANTA), 1)

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


def train_timestamp_model():
    model = TimestampModel()
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.summary()

    # Preprocess the data
    TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = preprocess_split("beatmaps")

    TRAIN_Y = np.array(
        list(
            map(lambda x: list(map(lambda y: 1
                                   if y[2] > 0 else 0, x)), TRAIN_Y)))
    TEST_Y = np.array(
        list(
            map(lambda x: list(map(lambda y: 1
                                   if y[2] > 0 else 0, x)), TEST_Y)))

    # Train the model
    model.fit(TRAIN_X,
              TRAIN_Y,
              epochs=10,
              batch_size=32,
              validation_data=(TEST_X, TEST_Y))

    # Save the model
    model.save("timestamp.keras")


@keras.saving.register_keras_serializable(package="convmusic")
class PositionModel(Model):

    def __init__(self):
        super(PositionModel, self).__init__()

        # Concatenate db list + encoded time
        INPUT_SHAPE = (2 * int(MAX_SONG_LENGTH / TIME_QUANTA), )

        self.model = Sequential([
            Input(shape=INPUT_SHAPE),
            Dense(128, activation='linear'),
            Dense(OUTPUT_SHAPE[1] * 2, activation='linear')
        ])

    def call(self, inputs):  # type: ignore
        print(inputs.shape)
        return self.model(inputs)

    def get_config(self):
        return {}


def train_position_model():
    model = PositionModel()
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # Preprocess the data
    TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = preprocess_split("beatmaps")

    NEW_TRAIN_X = []
    for i in range(len(TRAIN_X)):
        current_x = TRAIN_X[i]
        current_y = TRAIN_Y[i]
        sample = []
        for x, y in zip(current_x, current_y):
            sample += [x, 1 if y[2] > 0 else 0]
        NEW_TRAIN_X.append(sample)
    TRAIN_X = np.array(NEW_TRAIN_X)

    NEW_TEST_X = []
    for i in range(len(TEST_X)):
        current_x = TEST_X[i]
        current_y = TEST_Y[i]
        sample = []
        for x, y in zip(current_x, current_y):
            sample += [x, 1 if y[2] > 0 else 0]
        NEW_TEST_X.append(sample)
    TEST_X = np.array(NEW_TEST_X)

    NEW_TRAIN_Y = []
    for i in range(len(TRAIN_Y)):
        current_y = TRAIN_Y[i]
        sample = []
        for y in current_y:
            sample += [y[0], y[1]]
        NEW_TRAIN_Y.append(sample)
    TRAIN_Y = np.array(NEW_TRAIN_Y)

    NEW_TEST_Y = []
    for i in range(len(TEST_Y)):
        current_y = TEST_Y[i]
        sample = []
        for y in current_y:
            sample += [y[0], y[1]]
        NEW_TEST_Y.append(sample)
    TEST_Y = np.array(NEW_TEST_Y)

    print(TRAIN_X[0])
    print(TRAIN_X[0][0:200])

    # Train the model
    model.fit(TRAIN_X,
              TRAIN_Y,
              epochs=10,
              batch_size=32,
              validation_data=(TEST_X, TEST_Y))

    # Save the model
    model.save("position.keras")


if __name__ == "__main__":
    import sys
    import os
    from pydub import AudioSegment

    if "--train" in sys.argv:
        # train_timestamp_model()
        train_position_model()
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
