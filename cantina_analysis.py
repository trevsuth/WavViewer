import librosa
import librosa.display
import soundfile as sf

audio_path = './WavFiles/CantinaBand60.wav'
# load audio
y, sr = librosa.load(audio_path)
# y = numpy array of amplitudes
# sr = sampling rate.  SR is number of samples taken per second

"""# extract features
mfccs = librosa.feature.mfcc(y=y, sr=sr)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
"""

# ALt way to calculate phss
# Compute the short-time Fourier transform (STFT)
D = librosa.stft(y)
# Separate the harmonic and percussive components
H, P = librosa.decompose.hpss(D)
# Convert back to time domain
harmonic = librosa.istft(H)
percussive = librosa.istft(P)
#plt.plot(harmonic)
#plt.plot(percussive)
#plt.show()

sf.write('./harmonic.wav', harmonic, sr)
sf.write('./percussive.wav', percussive, sr)