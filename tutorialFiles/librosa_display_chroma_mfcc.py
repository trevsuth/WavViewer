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

mfcc = librosa.feature.mfcc(y=y, 
                            sr=sr, 
                            hop_length=hop_length,
                            n_mfcc=13)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
chroma = librosa.display.specshow(chromagram, 
                               y_axis='log', 
                               x_axis='time',
                               sr=sr, 
                               ax=ax[0])
mf = librosa.display.specshow(mfcc, 
                              y_axis='log',
                              x_axis='time',
                              ax=ax[1])
ax[0].set(title='Chromagram')
ax[0].label_outer()
ax[1].set(title='MFCC coefficients')
ax[1].label_outer()
plt.show()