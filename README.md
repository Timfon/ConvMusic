# Osu Mapper
## Team members: Timothy Fong, Michael Rehmet, Bokai Bi, Edward Wibowo

## Description:
Contains:
- A model that trains on a directory of beatmaps of your choosing.
- Contains a sequential model that uses 3 dense layers, with leaky_relu activation. 
- Uses mean absolute error to train the model on the mp3 timestamps. 
- Uses a custom mean squared error to train the model on the osu x and y coordinate positions. 

## Example usage:
- Create a beatmaps/ directory in the same directory as model.py. 
- Populate the directory with the osu beatmaps you want to train the model on. 
- type "python model.py --train" in the terminal.
- Get an .mp3 file of your choosing and place it in the same directory as model.py. 
- type "python model.py <song_name.mp3>"
- The model will output a .osz file which you can then play in osu!
- Have fun!

Estimated time to complete: >16.0 hours.