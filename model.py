from keras.layers import Input, LSTM, Conv1D, BatchNormalization, Dropout, Dense, Flatten  # type: ignore
from keras.models import Model, Sequential  # type: ignore
from preprocess import preprocess_split, MAX_SONG_LENGTH, TIME_QUANTA
import tensorflow as tf  # type: ignore
import numpy as np

OUTPUT_SHAPE = (None, int(MAX_SONG_LENGTH / TIME_QUANTA))
DROPOUT_RATIO = 0.1


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


class PositionModel(Model):

    def __init__(self):
        super(PositionModel, self).__init__()

        # Concatenate db list + encoded time
        INPUT_SHAPE = (2 * int(MAX_SONG_LENGTH / TIME_QUANTA), )

        self.model = Sequential([
            Input(shape=INPUT_SHAPE),
            Dense(256, activation='leaky_relu'),
            Dense(128, activation='leaky_relu'),
            Dense(OUTPUT_SHAPE[1] * 2, activation='leaky_relu')
        ])

    def call(self, inputs):  # type: ignore
        return self.model(inputs)


def train_position_model():
    model = PositionModel()
    model.compile(optimizer='adam', loss='log_cosh')
    model.summary()

    # Preprocess the data
    TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = preprocess_split("beatmaps")

    # TRAIN_X = [
    #            [db1, db2, ..., db4000],
    #            [db1, db2, ..., db4000],
    #           ]

    # TRAIN_X = [
    #            [db1, t1, db2, t2, ..., db4000, t4000],
    #            [db1, t1, db2, t2, ..., db4000, t4000],
    #           ]

    NEW_TRAIN_X = []
    for i in range(len(TRAIN_X)):
        current_x = TRAIN_X[i]
        current_y = TRAIN_Y[i]
        sample = []
        for x, y in zip(current_x, current_y):
            sample += [x, y[2]]
        NEW_TRAIN_X.append(sample)
    TRAIN_X = np.array(NEW_TRAIN_X)

    NEW_TEST_X = []
    for i in range(len(TEST_X)):
        current_x = TEST_X[i]
        current_y = TEST_Y[i]
        sample = []
        for x, y in zip(current_x, current_y):
            sample += [x, y[2]]
        NEW_TEST_X.append(sample)
    TEST_X = np.array(NEW_TEST_X)

    # TRAIN_Y = [
    #            [T1, T2, ..., T4000],
    #            [T1, T2, ..., T4000],
    #           ]
    # Ti = (xi, yi, ti)

    # TRAIN_Y = [
    #           [x1, y1, x2, y2, ..., x4000, y4000],
    #           [x1, y1, x2, y2, ..., x4000, y4000],
    #       ]

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

    # train_y_timestamp = np.array(
    # list(
    # map(lambda x: list(map(lambda y: 1
    # if y[2] > 0 else 0, x)), TRAIN_Y)))
    # test_y_timestamp = np.array(
    # list(
    # map(lambda x: list(map(lambda y: 1
    # if y[2] > 0 else 0, x)), TEST_Y)))

    # train_y_pos = np.array(
    # list(map(lambda x: list(map(lambda y: y[0:2], x)), TRAIN_Y)))
    # test_y_pos = np.array(
    # list(map(lambda x: list(map(lambda y: y[0:2], x)), TEST_Y)))

    # TRAIN_Y = np.concatenate((train_y_pos, train_y_timestamp), axis=1)
    # TEST_Y = np.concatenate((test_y_pos, test_y_timestamp), axis=1)

    print(TRAIN_X.shape)
    print(TRAIN_Y.shape)
    print(TEST_X.shape)
    print(TEST_Y.shape)

    # Train the model
    model.fit(TRAIN_X,
              TRAIN_Y,
              epochs=10,
              batch_size=32,
              validation_data=(TEST_X, TEST_Y))

    # Save the model
    model.save("position.keras")

    predictions = model.predict(TEST_X)

    print("predicted: ", predictions[0])
    print("expected: ", TEST_Y[0])
    print("predicted: ", sum(predictions[0]))
    print(
        "predicted: ",
        sum(list(map(lambda x: 0
                     if x < 0 else round(x), list(predictions[0])))))
    print("expected: ", sum(TEST_Y[0]))


train_position_model()
