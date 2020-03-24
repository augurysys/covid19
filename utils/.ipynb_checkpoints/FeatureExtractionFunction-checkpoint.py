# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:53:19 2020

@author: Daniel Teitelman
"""

# %% Requirements

'''
In order to run FeatureExtractionFunction.py follow the follwing instructions:
    1. In console: pip install librosa.
    2. Download Praat from the follwing link: http://www.fon.hum.uva.nl/praat/download_win.html no need to install Phonetic and international symbols or anything else.
    3. Unzip the zip you downloaded and place praat.exe in your desktop.
    4. In console: pip install praat-parselmouth.
    5. In console: pip install nolds. !!optional!! for chaos theory based features dont install it at first. In addition you might need an additional library named quantumrandom.
    6. Profit.
'''

# %% Imports

import numpy as np
import scipy
import librosa
import parselmouth
from scipy.io import wavfile
from parselmouth.praat import call
import nolds

# %% Functions

# sound here is 8 bit uint

def get_entropy(signal): 
    hist1 = np.histogram(signal[:,0],np.max(signal))
    hist2 = np.histogram(signal[:,1],np.max(signal))
    hist1_dist = scipy.stats.rv_histogram(hist1).pdf(np.linspace(0,np.max(signal),np.max(signal)+1))
    hist2_dist = scipy.stats.rv_histogram(hist2).pdf(np.linspace(0,np.max(signal),np.max(signal)+1))
    entropyLeftChannel = scipy.stats.entropy(hist1_dist)
    entropyRightChannel = scipy.stats.entropy(hist2_dist)
    return entropyLeftChannel, entropyRightChannel

# sound  = .wav sound file
# sr = sample rate usually for wav its 44.1k , sr is extracted by scipy or librosa

def get_features(sound_lib,sr,sound_scipy,sound_praat):
    features = []
    
    # Features extracted using parselmouth and pratt
    f0min = 75; f0max = 500;                                                            # Limits of human speach in Hz
    pitch = call(sound_praat, "To Pitch", 0.0, f0min, f0max)                            # create a praat pitch object
    harmonicity = call(sound_praat, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)          # create a praat harmonicity object
    pointProcess = call(sound_praat, "To PointProcess (periodic, cc)", f0min, f0max)    # create a praat pointProcess object
    unit = "Hertz"
    
    features.append(call(pitch, "Get mean", 0, 0, unit))                                                        # F0 - Central Frequency
    features.append(call(pitch, "Get standard deviation", 0 ,0, unit))                                          # F0 - std
    features.append(call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))                          # Relative jitter 
    features.append(call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3))                # Absolute jitter
    features.append(call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))                            # Relative average perturbation
    features.append(call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3))                           # 5-point period pertubation quotient ( ppq5 )
    features.append(call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3))                            # Difference of differences of periods ( ddp )
    features.append(call([sound_praat, pointProcess], "Get shimmer (local)" , 0, 0, 0.0001, 0.02, 1.3, 1.6))    # Relative Shimmer
    features.append(call([sound_praat, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6))  # Relative Shimmer dB
    features.append(call([sound_praat, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6))      # Shimmer (apq3)
    features.append(call([sound_praat, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6))      # Shimmer (apq5)
    features.append(call([sound_praat, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6))     # Shimmer (apq11)
    features.append(call([sound_praat, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6))       # Shimmer (dda)
    features.append(call(harmonicity, "Get mean", 0, 0))                                                        # Harmonic Noise Ratio 
    
    
    # Features extracted using librosa
    features.append(librosa.feature.spectral_flatness(sound_lib))           # Spectral Flatness
    features.append(librosa.feature.rms(sound_lib))                         # Volume
    features.append(librosa.feature.zero_crossing_rate(sound_lib))          # Zero Crossing Rate
    features.append(librosa.feature.spectral_centroid(sound_lib,sr))        # Spectral Centroind
    features.append(librosa.feature.spectral_bandwidth(sound_lib,sr))       # Spectral Bandwidth
    features.append(librosa.feature.spectral_contrast(sound_lib,sr))        # Spectral Contrast
    features.append(librosa.feature.spectral_rolloff(sound_lib,sr))         # Spectral Rolloff
    features.append(librosa.feature.mfcc(sound_lib,sr))                     # Mel-Frequency Cepstral Coefficients - MFCC 
    features.append(librosa.feature.tonnetz(sound_lib,sr))                  # Tonnetz
    features.append(librosa.feature.chroma_stft(sound_lib,sr))              # Spectrogram
    features.append(librosa.feature.chroma_cqt(sound_lib,sr))               # Constant-Q Chromagram
    features.append(librosa.feature.chroma_cens(sound_lib,sr))              # Chroma Energy Normalized
    
    # tempogram feature might be useless as it is too redundant - un comment it if you find it usefull
    #features.append(librosa.feature.tempogram(sound_lib,sr))               # Tempogram: local autocorrelation of the onset strength envelope 
    
    # Features extracted using scipy
    features.append(scipy.stats.skew(sound_lib))                            # Skewness
    entropy = get_entropy(sound_scipy)
    features.append(entropy[0])                                             # Entropy Left Channel
    features.append(entropy[1])                                             # Entropy Right Channel
    
    # Features extracted by nolds (Chaos/Dynamical Systems Theory) - comment this if you didnt download nolds
    features.append(nolds.hurst_rs(sound_lib))                              # The hurst exponent is a measure of the “long-term memory” of a time series
    
    # Please dont use this even if you downloaded nolds
    # The following features require extremely long computation time and dont run by normal means, please dont use them to save yourself from having a headache. #I cant gurantite they will even coverage (dependents on the leangth of the audio file)#
    #features.append(nolds.dfa(sound_lib))                                  # Performs a detrended fluctuation analysis (DFA) on the given data
    #features.append(nolds.lyap_r(sound_lib))                               # Estimates the largest Lyapunov exponent using the algorithm of Rosenstein
    #features.append(nolds.lyap_e(sound_lib))                               # Estimates the Lyapunov exponents for the given data using the algorithm of Eckmann
    #features.append(nolds.corr_dim(sound_lib,1))                           # Calculates the correlation dimension with the Grassberger-Procaccia algorithm
    
    return features
    
# %% Main

# Important!! read files in the following way:
   
if __name__ == "__main__":
    directory = r"C:\Users" # your directory of choice
    data_praat = parselmouth.Sound(directory + "\example.wav")
    fs_scipy, data_scipy = wavfile.read(directory + "\example.wav") # Audio read by the wavfile.read function from scipy has both left channel and right channel data inside of it. Where data[:, 0] is the left channel and data[:, 1] is the right channel.
    data_librosa = librosa.load(directory + "\example.wav", sr=44100)
    Features = get_features(data_librosa[0],data_librosa[1],data_scipy,data_praat)