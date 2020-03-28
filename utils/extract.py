# imports:

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# config:

f_s = 44100 # [Hz]
# f_s = 48000 # [Hz]

n_fft = 4096
hop_length = 256

default = {
    'mag_a_min': 1e-13, # 1e-5, #
    'mag_top_db': 70.0, # 80.0, #
    'f_low_cut': 0, # 100, # 50, # 150 # 
    'f_high_cut': np.inf, # 5000, # 
    'n_ceps_coeffs': 300,
    'idx_ceps_low_cut': 0,
}

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

# params['cqt'] = {
#     'f_s': f_s, 
#     'n_bins': int(n_fft/2),
#     'hop_length': hop_length,
#     'fmin': librosa.core.note_to_hz('E0')
#     'bins_per_octave': 256,
#     'filter_scale': 1/8,
# }

params['cqt'] = {
    'f_s': f_s, 
    'n_bins': int((84+12)*4),
    'hop_length': hop_length,
    'fmin': librosa.core.note_to_hz('C1'), # = 32.70 [Hz]
    'bins_per_octave': int(12*4),
    'filter_scale': 1,
}
