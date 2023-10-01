import librosa

filename = './WavFiles/Fanfare60.wav'

#Load audo as a waveform y
#Store sampling rate as sr
y, sr = librosa.load(filename)

#here y is a timeseries represesented as a 
# 1 dimentional numpy array
#sr is the num of samples per second of audio 
#default sr=22050 Hz, can by overridden via arguments
# to librosa.load()

#Get tempo, beat
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#here beat times is an array of the timestamp 
# (in seconds) of the beats

print('Estimated temp: {:2f} bpm'.format(tempo))

#convert the frame indices of beat events into timestamps
beat_time = librosa.frames_to_time(beat_frames, sr=sr)
print('beat_time:')
print(beat_time)