"""
Preprocesses beatmaps in .osz format.
"""
import numpy as np
import os
import zipfile
import tempfile
import concurrent.futures
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

N_FFT = 512
TIME_QUANTA = 30  # miliseconds
MAX_SONG_LENGTH = 1000 * 120


def extract_decibels(sound, quanta=TIME_QUANTA):
    """
    Returns a list of average interval decibels for the given file, each interval is quanta long
    Song is truncated to MAX_SONG_LENGTH
    """

    sound = sound[:MAX_SONG_LENGTH]
    output = []
    start_ms = 0
    while (start_ms + quanta <= MAX_SONG_LENGTH):
        if (start_ms < len(sound)):
            output.append(max(-100, sound[start_ms:start_ms + quanta].dBFS))
        else:
            output.append(
                -100
            )  # -inf is used as padding since pydub db is relative; 0db is the maximum loudness
        start_ms += quanta
    return np.array(output)


def vectorize_hit_objects(filename, quanta=TIME_QUANTA):
    with open(filename, "r", encoding="utf8") as f:
        content = f.read()

    start = content.find("[HitObjects]")
    assert start != -1, "No hit objects found in the file"

    section = content[start:].split("\n")[1:]

    hit_objects = []

    for line in section:
        if line.strip() == "":
            continue

        elements = line.split(",")
        x = int(elements[0])
        y = int(elements[1])
        time = int(elements[2])

        # Convert sliders to hit circles
        hit_objects.append((x, y, time))
    output = []
    start_ms = 0
    while (start_ms + quanta <= MAX_SONG_LENGTH):
        obj = None
        while (len(hit_objects) > 0 and hit_objects[0][2] >= start_ms
               and hit_objects[0][2] < start_ms + quanta):
            obj = hit_objects.pop(0)
        output.append(obj if obj else (0, 0, 0))
        start_ms += quanta
    return np.array(output)


def process_beatmap(beatmap, beatmap_dir):
    """
    Processes the given beatmap file in the given directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        osz_file = os.path.join(beatmap_dir, beatmap)
        with zipfile.ZipFile(osz_file, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        osu_file, audio_file = None, None
        for file in os.listdir(temp_dir):
            if file.endswith(".osu"):
                osu_file = os.path.join(temp_dir, file)
            elif (file.endswith(".mp3") or file.endswith(".ogg")) and not (
                    "hitwhistle" in file or "hitnormal" in file or "hitclap"
                    in file or "hitfinish" in file or "sliderslide" in file):
                audio_file = os.path.join(temp_dir, file)

        # Ensure both files were found
        if osu_file and audio_file:
            assert os.path.isfile(audio_file)
            sound = AudioSegment.empty()
            if (audio_file[-3:] == "ogg"):
                sound = AudioSegment.from_ogg(audio_file)
            elif (audio_file[-3:] == "mp3"):
                sound = AudioSegment.from_mp3(audio_file)
            else:
                raise Exception(
                    f'Unknown file ending {audio_file[-3:]} encountered in extract decibels'
                )

            decibels = extract_decibels(sound)
            hit_object_vector = vectorize_hit_objects(osu_file)
            return decibels, hit_object_vector
        else:
            return None, None  # or handle missing files as needed


def preprocess(beatmap_dir, max_workers=4):
    """
    Preprocesses the beatmaps in the given directory.
    Each beatmap should be in .osz format.
    """
    decibels = []
    hit_objects = []
    beatmaps = [
        beatmap for beatmap in os.listdir(beatmap_dir)
        if beatmap.endswith(".osz")
    ]

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as executor:
        future_to_beatmap = {
            executor.submit(process_beatmap, beatmap, beatmap_dir): beatmap
            for beatmap in beatmaps
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_beatmap)):
            result = future.result()
            if result is not None:
                decibels.append(result[0])
                hit_objects.append(result[1])

    return np.array(decibels), np.array(hit_objects)


def preprocess_split(beatmap_dir, split=0.8):
    """
    Preprocesses the beatmaps in the given directory and splits them into training and test sets.
    """
    X, Y = preprocess(beatmap_dir)

    TRAIN_X, TRAIN_Y = X[:int(split * len(X))], Y[:int(split * len(Y))]
    TEST_X, TEST_Y = X[int(split * len(X)):], Y[int(split * len(Y)):]
    assert len(TRAIN_X) == len(TRAIN_Y)
    assert len(TEST_X) == len(TEST_Y)
    return TRAIN_X, TRAIN_Y, TEST_X, TEST_Y
