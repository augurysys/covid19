# imports:

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# config:

f_s = 44100 # [Hz]
# f_s = 48000 # [Hz]

default = {
    'mag_a_min': 1e-13, # 1e-5, #
    'mag_top_db': 70.0, # 80.0, #
    'f_low_cut': 150, # 0, # 50, # 
    'f_high_cut': 5000, # np.inf, # 
    'n_ceps_coeffs': 300,
    'idx_ceps_low_cut': 2,
}

n_fft = 4096
hop_length = 256

params = {}

params['stft'] = {
    'f_s': f_s, 
    'n_fft': n_fft,
    'win_length': n_fft, 
    'hop_length': hop_length, 
}

params['mel'] = {
    'f_s': f_s, 
    'n_fft': n_fft,
    'hop_length': hop_length,
    'fmin': 0,
    'fmax': 5259.7802, # f_s/2, #
    'n_mels': 256, 
}

params['cqt'] = {
    'f_s': f_s, 
    'n_bins': int(n_fft/2),
    'hop_length': hop_length,
    'fmin': librosa.core.note_to_hz('E0'), # = 32.70 [Hz]
    'bins_per_octave': 256,
    'filter_scale': 1/8,
}
