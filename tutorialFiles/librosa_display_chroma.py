# https://librosa.org/doc/latest/generated/librosa.display.specshow.html#librosa.display.specshow

import numpy as np
import matplotlib.pyplot as plt
import librosa

# file to load
filename = 'WavFiles/Fanfare60.wav'

# load the file
y, sr = librosa.load(filename)

# Set the hop length; @sr=22050Hz, 512 ~= 23ms
hop_length = 512

# Seperate the harmonics and percussives into 2 wavforms
y_harmonic, y_percussive = librosa.effects.hpss(y)

# make chromagram from harmonic signal
chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)

fig, ax = plt.subplots()
img = librosa.display.specshow(chromagram, 
                               y_axis='linear', 
                               x_axis='time',
                               sr=sr, 
                               ax=ax)
plt.show()