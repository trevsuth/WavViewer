# more librosa stuff
# https://librosa.org/doc/latest/tutorial.html

import numpy as np
import librosa

# file to load
filename = 'WavFiles/Fanfare60.wav'

# load the file
y, sr = librosa.load(filename)

# Set the hop length; @sr=22050Hz, 512 ~= 23ms
hop_length = 512

# Seperate the harmonics and percussives into 2 wavforms
# using harmonic-precussive seperation series
y_harmonic, y_percussive = librosa.effects.hpss(y)


# Get the beat track from the percussive signal
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                             sr=sr)

# compute MFCC features from raw signal
# MFCC = mel-frequency cepstral coefficients
mfcc = librosa.feature.mfcc(y=y, 
                            sr=sr, 
                            hop_length=hop_length,
                            n_mfcc=13)
# output is a numpy.ndarray in shape [n_mfcc, track duration in frames]
# hop_length = hop_length ensures that data will be synced 
# w/ beat tracker



# compute forst order differences
mfcc_delta = librosa.feature.delta(mfcc)

# stack and sync beat b/t beat events
# use mean valus instead of median
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                    beat_frames)

# compute chroma features from harmonic signal
chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)

# agg chroma features between beat events
beat_chroma = librosa.util.sync(chromagram,

                                beat_frames,
                                aggregate=np.median)

# stick all the beat-sync features together
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])