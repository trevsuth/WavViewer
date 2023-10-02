import librosa
import matplotlib.pyplot as plt

filename = './WavFiles/Fanfare60.wav'

y, sr = librosa.load(filename)


#Get tempo, beat
#tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

plt.plot(y)
plt.show()