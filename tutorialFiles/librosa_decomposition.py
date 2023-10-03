import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition
import librosa

# file to load
filename = 'WavFiles/StarWars60.wav'

# load the file
y, sr = librosa.load(filename)
S = np.abs(librosa.stft(y))

#sort by ascending path frequenct
comps, acts = librosa.decompose.decompose(S, 
                                          n_components=16,
                                          sort=True)
#or with sparse dictionary learning
T = sklearn.decomposition.MiniBatchDictionaryLearning(n_components=16,
                                                      max_iter=1000,
                                                      alpha=0.1)
scomps, sacts = librosa.decompose.decompose(S, 
                                            transformer=T, 
                                            sort=True)
layout = [list(".AAAA"), 
          list("BCCCC"), 
          list(".DDDD")]
fig, ax = plt.subplot_mosaic(layout, 
                             constrained_layout=True)

librosa.display.specshow(librosa.amplitude_to_db(S, 
                                                 ref=np.max),
                         y_axis='log', 
                         x_axis='time', 
                         ax=ax['A'])
ax['A'].set(title='Input spectrogram')
ax['A'].label_outer()
librosa.display.specshow(librosa.amplitude_to_db(comps,
                                                 ref=np.max),
                         y_axis='log', 
                         ax=ax['B'])
ax['B'].set(title='Components')
ax['B'].label_outer()
ax['B'].sharey(ax['A'])
librosa.display.specshow(acts, x_axis='time', ax=ax['C'], cmap='gray_r')
ax['C'].set(ylabel='Components', title='Activations')
ax['C'].sharex(ax['A'])
ax['C'].label_outer()
S_approx = comps.dot(acts)
img = librosa.display.specshow(librosa.amplitude_to_db(S_approx,
                                                       ref=np.max),
                               y_axis='log', 
                               x_axis='time', 
                               ax=ax['D'])
ax['D'].set(title='Reconstructed spectrogram')
ax['D'].sharex(ax['A'])
ax['D'].sharey(ax['A'])
ax['D'].label_outer()
fig.colorbar(img, 
             ax=list(ax.values()), 
             format="%+2.f dB")