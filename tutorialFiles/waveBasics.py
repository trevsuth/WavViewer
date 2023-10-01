#an intro to the wave library

import wave

# Open a WAV file
with wave.open('./WavFiles/ImperialMarch60.wav', 'rb') as file:
    n_channels = file.getnchannels()
    sampwidth = file.getsampwidth()
    framerate = file.getframerate()
    n_frames = file.getnframes()
    frames = file.readframes(n_frames)

print('n_channels: {}'.format(n_channels) )
print('sampwidth: {}'.format(sampwidth))
print('framerate: {}' .format(framerate))
#print(n_frames)
#print(frames)