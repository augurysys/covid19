# imports:

import librosa
import scipy
import matplotlib.pyplot as plt
import numpy as np

import utils_aug.extract
import utils_aug.trans


# functions:

def signal_features(
    x, normalize=True,
    f_s=utils_aug.extract.f_s, 
    n_fft=utils_aug.extract.n_fft,
    frame_length=utils_aug.extract.n_fft,
    hop_length=utils_aug.extract.hop_length):
    
    # input data:
    wave = x['wave']['s']
    spec = x['fft']['s']

    # padding and windowing:
    wave_padded = np.pad(wave, int(frame_length//2), mode='reflect')
    wave_windows = librosa.util.frame(wave_padded, frame_length=frame_length, hop_length=hop_length)
    
    features = {'rms': librosa.feature.rms(wave, hop_length=hop_length, frame_length=frame_length),
               'rms_total': librosa.feature.rms(wave),
               'spectral_flatness' : librosa.feature.spectral_flatness(wave, hop_length=hop_length, n_fft=n_fft),
               'std' : np.std(wave, axis=0),
               'zero_cross':librosa.feature.zero_crossing_rate(wave, hop_length=hop_length, frame_length=frame_length)}

    return features
    
    # features extraction using librosa:
#     features_1 = []
     #features_1.append(librosa.feature.spectral_flatness(wave, hop_length=hop_length, n_fft=n_fft)) 
#     features_1.append(librosa.feature.rms(wave, hop_length=hop_length, frame_length=frame_length)) 
#     features_1.append(librosa.feature.zero_crossing_rate(wave, hop_length=hop_length, frame_length=frame_length))     
#     features_1.append(librosa.feature.spectral_centroid(wave, f_s, hop_length=hop_length, n_fft=n_fft))
#     features_1.append(librosa.feature.spectral_bandwidth(wave, f_s, hop_length=hop_length, n_fft=n_fft))   
#     features_1.append(librosa.feature.spectral_contrast(wave, f_s, hop_length=hop_length, n_fft=n_fft)) 
#     features_1.append(librosa.feature.spectral_rolloff(wave, f_s, hop_length=hop_length, n_fft=n_fft)) 
#     features_1.append(librosa.feature.mfcc(wave, f_s, hop_length=hop_length, n_fft=n_fft))
#     features_1.append(librosa.feature.chroma_stft(wave, f_s, hop_length=hop_length, n_fft=n_fft))

#     # features extraction using scipy:
#     features_2 = []
#     features_2.append(np.std(wave_windows, axis=0))
#     features_2.append(scipy.stats.skew(wave_windows, axis=0))
#     features_2.append(scipy.stats.kurtosis(wave_windows, axis=0))
#     features_2.append(np.stack([scipy.stats.entropy(np.abs(w+1e-10)) for w in wave_windows.T]))

#     for i in range(len(features_2)):
#         features_2[i] = np.expand_dims(features_2[i], axis=0)

#     # features extraction from spectrograms:
#     features_3 = []
#     features_3.append(np.std(spec, axis=0))
#     features_3.append(scipy.stats.skew(spec, axis=0))
#     features_3.append(scipy.stats.kurtosis(spec, axis=0))
#     features_3.append(np.stack([scipy.stats.entropy(s+1e-10) for s in spec.T]))
    
#     for i in range(len(features_2)):
#         features_3[i] = np.expand_dims(features_3[i], axis=0)

#     # concatenate:
# #     features_1 = np.concatenate(features_1)
# #     features_2 = np.concatenate(features_2)
# #     features_3 = np.concatenate(features_3)
# #     features = np.concatenate([features_1, features_2, features_3])

# #     # normalize: (OPTIONAL, not sure if releval)
# #     if normalize:
# #         for i in range(len(features)):
# #             features[i] = (features[i] - np.mean(features[i]))/np.std(features[i])

   


