import librosa
import matplotlib.pyplot as plt

audio_path = './WavFiles/CantinaBand60.wav'
# load audio
y, sr = librosa.load(audio_path)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(tempo) # in beats per min
print(beat_frames) #
# print('Estimated tempo: {:.2f} beats per minute'.format(tempo[0]))

beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print(beat_times)