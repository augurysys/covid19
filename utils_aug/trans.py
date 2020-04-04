# imports:

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy

from utils_aug.extract import default


# wavelet functions:

def wave_to_stft(
    wave, 
    f_s, win_length, 
    hop_length, n_fft):
    
    # short-time fourier transform:
    stft = librosa.stft(wave, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    stft_spec, stft_phase = librosa.magphase(stft)
    
    # use magnitude and phase:
    stft_spec_db = scale_magnitude(stft_spec)
    stft_f_phase = f_phase(stft_phase)
    
    # create cepstrum:
    stft_ceps = cepstrum(stft_spec_db)
    
    # create axes:
    n_f = stft.shape[0]
    n_t = stft.shape[1]
    max_t = (len(wave)-1)/f_s
    f_stft = np.linspace(0, f_s/2, n_f)
    t_stft = np.linspace(0, max_t, n_t)
    
    # organize:
    stft = {
        's': stft_spec_db,
        'c': stft_ceps,
        'f_phase': stft_f_phase,
        'f': f_stft, # [Hz]
        't': t_stft,
    }
    
    # filter frequency:
    stft = band_pass_filter(stft)
    
    # re-scale frequency:
    stft['f'] = 1e-3*stft['f']
    
    return stft


def wave_to_mel(
    wave, 
    f_s, n_fft, hop_length,
    fmin, fmax, n_mels):
    
    # transform - mel spectrogram:
    mel_spec = librosa.feature.melspectrogram(
        wave, 
        n_fft=n_fft, hop_length=hop_length, 
        n_mels=n_mels, sr=f_s, fmin=fmin, fmax=fmax, 
        power=1)
    
    mel_spec_db = scale_magnitude(mel_spec)
#     mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    
    # create cepstrum (mfcc):
    mel_ceps = cepstrum(mel_spec_db)
    
    # create axes:
    max_t = (len(wave)-1)/f_s
    n_t = mel_spec.shape[1]
    f_mel = librosa.core.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False)
    t_mel = np.linspace(0, max_t, n_t)
    
    # organize:
    mel = {
        's': mel_spec_db,
        'c': mel_ceps,
        'f': f_mel, # [Hz]
        't': t_mel
    }
    
    # filter frequency:
    mel = band_pass_filter(mel)
    
    # re-scale frequency:
    mel['f'] = 1e-3*mel['f']
    
    return mel


def wave_to_cqt(
    wave, 
    f_s, hop_length, 
    fmin, n_bins, bins_per_octave, 
    filter_scale):
    
    # transform - cqt spectrogram:
    cqt = librosa.core.cqt(
        wave, 
        sr=f_s, hop_length=hop_length, 
        fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, 
        filter_scale=filter_scale, norm=1)
    cqt_spec, cqt_phase = librosa.magphase(cqt)
    
    # use magnitude and phase:
    cqt_spec_db = scale_magnitude(cqt_spec)
    cqt_f_phase = f_phase(cqt_phase)
    
    # create cepstrum (cqcc):
    cqt_ceps = cepstrum(cqt_spec_db)
    
    # create axes:
    max_t = (len(wave)-1)/f_s
    n_t = cqt_spec.shape[1]
    f_cqt = librosa.core.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    t_cqt = np.linspace(0, max_t, n_t)
    
    # organize:
    cqt = {
        's': cqt_spec_db,
        'c': cqt_ceps,
        'f_phase': cqt_f_phase,
        'f': f_cqt, # [Hz]
        't': t_cqt
    }
    
    # filter frequency:
    cqt = band_pass_filter(cqt)
    
    # re-scale frequency:
    cqt['f'] = 1e-3*cqt['f']
    
    return cqt


# wrapper functions:

def apply(x_wave, func, func_params):
    
    x_spec = func(x_wave['s'], **func_params)
    
    return x_spec


def band_pass_filter(
    x_spec, 
    f_low_cut=default['f_low_cut'], 
    f_high_cut=default['f_high_cut']):
    
    f_idx_keep = (x_spec['f']>=f_low_cut) & (x_spec['f']<=f_high_cut)
    x_spec['s'] = x_spec['s'][f_idx_keep, :]
    x_spec['f'] = x_spec['f'][f_idx_keep]
    if 'f_phase' in x_spec.keys():
        x_spec['f_phase'] = x_spec['f_phase'][f_idx_keep, :]
    
    return x_spec


def f_phase(phase):
    
    phase_angle = np.angle(phase)
    phase_unwrapped = np.unwrap(phase_angle)
    d_phase = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
    f_phase = np.concatenate([phase_unwrapped[:, 0:1], d_phase], axis=1)/np.pi
    
    return f_phase


def scale_magnitude(
    mag,
    amin=default['mag_a_min'], 
    top_db=default['mag_top_db']):
    
    scaled_mag = 1+(librosa.amplitude_to_db(mag, ref=np.max, amin=amin, top_db=top_db)/top_db)
    
    return scaled_mag


def cepstrum(
    spectrum, 
    n_coeffs=default['n_ceps_coeffs'], 
    idx_low_cut=default['idx_ceps_low_cut']):
    # assuming axis=0 represents frequency (linear or other)

    ceps = scipy.fftpack.dct(spectrum, axis=0, type=2, norm='ortho')
    ceps = ceps[idx_low_cut:idx_low_cut+n_coeffs]
    
    return ceps


def wave_to_fft(wave,f_s):
    
    from scipy.fftpack import fft, ifft
    import numpy as np
    x=wave
    L=len(x)
    T=L/f_s
    sx=1/L*fft(x)
    
    s_fft=np.abs(sx[:int(L/2)])
    f_fft=np.linspace(0,f_s/2,f_s/2*T)
    t_fft=T
    
    # organize:
    fft = {
        's': s_fft,
        'f': f_fft, 
    }
    
    return fft