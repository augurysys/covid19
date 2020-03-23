# imports:

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# config:

# f_s = 44100 # [Hz]
f_s = 48000 # [Hz]

# common parameters:

n_fft = 4096
hop_length = 256
f_low_cut = 0 # 50 # 
f_high_cut = 5000 # np.inf # 



params = {}

params['stft'] = {
    'f_s': f_s, 
    'n_fft': n_fft,
    'win_length': n_fft, 
    'hop_length': hop_length, 
    'f_low_cut': f_low_cut,
    'f_high_cut': f_high_cut,
}

params['mel'] = {
    'f_s': f_s, 
    'n_fft': n_fft,
    'hop_length': hop_length,
    'fmin': 0,
    'fmax': 5259.7802, # f_s/2, #
    'n_mels': 256, 
    'f_low_cut': f_low_cut,
    'f_high_cut': f_high_cut,
}

params['cqt'] = {
    'f_s': f_s, 
    'n_bins': int(n_fft/2),
    'hop_length': hop_length,
    'fmin': librosa.core.note_to_hz('E0'), # = 32.70 [Hz]
    'bins_per_octave': 256,
    'filter_scale': 1/8,
    'f_low_cut': f_low_cut,
    'f_high_cut': f_high_cut,
}
    

# functions:

def apply(x_wave, func, func_params):
    
    x_spec = func(x_wave['s'], **func_params)
    
    return x_spec