"""
Preprocesses beatmaps in .osz format.
"""
import librosa
import numpy as np
import os
import zipfile
import tempfile
import concurrent.futures
import numpy as np
from keras.utils import pad_sequences  # type: ignore
from pydub import AudioSegment

N_FFT = 512
TIME_QUANTA = 30 # miliseconds
MAX_SONG_LENGTH = 1000 * 120

def extract_decibals(sound, quanta=TIME_QUANTA):
    """
    Returns a list of average interval decibals for the given file, each interval is quanta long
    Song is truncated to MAX_SONG_LENGTH
    """
    
    sound = sound[:MAX_SONG_LENGTH]
    output = []
    start_ms = 0
    while (start_ms + quanta <= MAX_SONG_LENGTH):
        if (start_ms < len(sound)):
            output.append(sound[start_ms:start_ms+quanta].dBFS)
        else:
            output.append(float('-inf')) # -inf is used as padding since pydub db is relative; 0db is the maximum loudness
        start_ms += quanta
    return np.array(output)


def vectorize_hit_objects(filename, quanta=TIME_QUANTA):
    """
    Vectorizes the hit objects in the given osu file.
    
    Returns a list of tuples, each tuple containing the following elements:
    1. x-coordinate
    2. y-coordinate
    3. time

    Note: Sliders are converted to hit circles.
    """
    with open(filename, "r", encoding="utf8") as f:
        content = f.read()

    start = content.find("[HitObjects]")
    assert start != -1, "No hit objects found in the file"

    section = content[start:].split("\n")[1:]

    vector = []

    for line in section:
        if line.strip() == "":
            continue

        elements = line.split(",")
        x = int(elements[0])
        y = int(elements[1])
        time = int(elements[2])

        # Convert sliders to hit circles
        vector.append((time))
    output = []
    start_ms = 0
    while (start_ms + quanta <= MAX_SONG_LENGTH):
        has_hit = False
        while (len(vector) > 0 and vector[0] >= start_ms and vector[0] < start_ms + quanta):
            has_hit = True
            vector.pop(0)
        output.append(1 if has_hit else 0)
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
            #print("-----------------------------------------Audio file ", audio_file)
            assert os.path.isfile(audio_file)
            sound = AudioSegment.empty()
            if (audio_file[-3:] == "ogg"):
                sound = AudioSegment.from_ogg(audio_file)
            elif (audio_file[-3:] == "mp3"):
                sound = AudioSegment.from_mp3(audio_file)
            else:
                raise("Unknown file ending ", audio_file[-3:], " encountered in extract decibals")
            
            decibals = extract_decibals(sound)
            hit_object_vector = vectorize_hit_objects(osu_file)
            return decibals, hit_object_vector
        else:
            return None, None  # or handle missing files as needed


def preprocess(beatmap_dir, max_workers=4):
    """
    Preprocesses the beatmaps in the given directory.
    Each beatmap should be in .osz format.
    """
    decibals = []
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

        for future in concurrent.futures.as_completed(future_to_beatmap):
            result = future.result()
            if result is not None:
                decibals.append(result[0])
                hit_objects.append(result[1])

    return np.array(decibals), np.array(hit_objects)


def preprocess_split(beatmap_dir, split=0.8):
    """
    Preprocesses the beatmaps in the given directory and splits them into training and test sets.
    """
    X, Y = preprocess(beatmap_dir)

    # Calculate max length of spectrogram and hit object vector
    #max_len = max(max([len(x) for x in X]), max([len(y) for y in Y]))

    # Pad the sequences
    #X = pad_sequences(X, padding='post', dtype='float32', maxlen=max_len)
    #Y = pad_sequences(Y, padding='post', dtype='int32', maxlen=max_len)

    TRAIN_X, TRAIN_Y = X[:int(split * len(X))], Y[:int(split * len(Y))]
    TEST_X, TEST_Y = X[int(split * len(X)):], Y[int(split * len(Y)):]
    assert len(TRAIN_X) == len(TRAIN_Y)
    assert len(TEST_X) == len(TEST_Y)
    return TRAIN_X, TRAIN_Y, TEST_X, TEST_Y
