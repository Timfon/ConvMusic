from preprocess import MAX_SONG_LENGTH, TIME_QUANTA
import os

def output_osu_string(timestamps, positions, audiofile_name, song_name):
    output_string = f'''osu file format v14

[General]
AudioFilename: {audiofile_name}
AudioLeadIn: 0
PreviewTime: 33312
Countdown: 0
SampleSet: Soft
StackLeniency: 0.4
Mode: 0
LetterboxInBreaks: 0
WidescreenStoryboard: 0

[Metadata]
Title:{song_name}
TitleUnicode:{song_name}
Artist:Unspecified
ArtistUnicode:Unspecified
Creator:ConvMusic
Version:Generated
Source:Internet
Tags:wow ai
BeatmapID:514514514

[Difficulty]
HPDrainRate:3.5
CircleSize:3.4
OverallDifficulty:5
ApproachRate:5.5
SliderMultiplier:0.999999999999999
SliderTickRate:1

[TimingPoints]
932,317.460317460317,3,2,2,20,1,0


[HitObjects]
'''
    
# object format: x,y,time,type(1),hitSound(0),hitSample(0:0:0:0:)
    assert(len(timestamps) == len(positions) / 2)
    assert(len(timestamps) == int(MAX_SONG_LENGTH / TIME_QUANTA))
    for quanta in range(0, int(MAX_SONG_LENGTH / TIME_QUANTA)):
        if (timestamps[quanta] == 0):
            continue
        new_hit_obj = f'{positions[quanta * 2]},{positions[quanta * 2 + 1]},{quanta*TIME_QUANTA},1,0,0:0:0:0:\n'
        output_string += new_hit_obj
    return output_string
    
def output_osu_file(osu_string, output_dir, song_name):
    path = os.path.join(output_dir, song_name)
    path += ".osu"
    with open(path, "w") as f:
        f.write(osu_string)
    