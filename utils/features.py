# imports:

import librosa
import scipy
import matplotlib.pyplot as plt
import numpy as np

import utils.extract
import utils.trans


# functions:

def signal_features(
    x, normalize=True,
    f_s=utils.extract.f_s, 
    n_fft=utils.extract.n_fft,
    frame_length=utils.extract.n_fft,
    hop_length=utils.extract.hop_length):
    
    # input data:
    wave = x['wave']['s']
    spec = x['mel']['s']

    # padding and windowing:
    wave_padded = np.pad(wave, int(frame_length//2), mode='reflect')
    wave_windows = librosa.util.frame(wave_padded, frame_length=frame_length, hop_length=hop_length)
    
    # features extraction using librosa:
    features_1 = []
    features_1.append(librosa.feature.spectral_flatness(wave, hop_length=hop_length, n_fft=n_fft)) 
    features_1.append(librosa.feature.rms(wave, hop_length=hop_length, frame_length=frame_length)) 
    features_1.append(librosa.feature.zero_crossing_rate(wave, hop_length=hop_length, frame_length=frame_length))     
    features_1.append(librosa.feature.spectral_centroid(wave, f_s, hop_length=hop_length, n_fft=n_fft))
    features_1.append(librosa.feature.spectral_bandwidth(wave, f_s, hop_length=hop_length, n_fft=n_fft))   
    features_1.append(librosa.feature.spectral_contrast(wave, f_s, hop_length=hop_length, n_fft=n_fft)) 
    features_1.append(librosa.feature.spectral_rolloff(wave, f_s, hop_length=hop_length, n_fft=n_fft)) 
    features_1.append(librosa.feature.mfcc(wave, f_s, hop_length=hop_length, n_fft=n_fft))
    features_1.append(librosa.feature.chroma_stft(wave, f_s, hop_length=hop_length, n_fft=n_fft))

    # features extraction using scipy:
    features_2 = []
    features_2.append(np.std(wave_windows, axis=0))
    features_2.append(scipy.stats.skew(wave_windows, axis=0))
    features_2.append(scipy.stats.kurtosis(wave_windows, axis=0))
    features_2.append(np.stack([scipy.stats.entropy(np.abs(w+1e-10)) for w in wave_windows.T]))

    for i in range(len(features_2)):
        features_2[i] = np.expand_dims(features_2[i], axis=0)

    # features extraction from spectrograms:
    features_3 = []
    features_3.append(np.std(spec, axis=0))
    features_3.append(scipy.stats.skew(spec, axis=0))
    features_3.append(scipy.stats.kurtosis(spec, axis=0))
    features_3.append(np.stack([scipy.stats.entropy(s+1e-10) for s in spec.T]))
    
    for i in range(len(features_2)):
        features_3[i] = np.expand_dims(features_3[i], axis=0)

    # concatenate:
    features_1 = np.concatenate(features_1)
    features_2 = np.concatenate(features_2)
    features_3 = np.concatenate(features_3)
    features = np.concatenate([features_1, features_2, features_3])

    # normalize: (OPTIONAL, not sure if releval)
    if normalize:
        for i in range(len(features)):
            features[i] = (features[i] - np.mean(features[i]))/np.std(features[i])

    return features


def lpc(y, order):
    # source: librosa
    # https://librosa.github.io/librosa/_modules/librosa/core/audio.html#lpc
  
    dtype = y.dtype.type
    ar_coeffs = np.zeros(order+1, dtype=dtype)
    ar_coeffs[0] = dtype(1)
    ar_coeffs_prev = np.zeros(order+1, dtype=dtype)
    ar_coeffs_prev[0] = dtype(1)

    fwd_pred_error = y[1:]
    bwd_pred_error = y[:-1]

    den = np.dot(fwd_pred_error, fwd_pred_error) \
          + np.dot(bwd_pred_error, bwd_pred_error)

    for i in range(order):
        if den <= 0:
            raise FloatingPointError('numerical error, input ill-conditioned?')

        reflect_coeff = dtype(-2) * np.dot(bwd_pred_error, fwd_pred_error) / dtype(den)

        ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
        for j in range(1, i + 2):
            ar_coeffs[j] = ar_coeffs_prev[j] + reflect_coeff * ar_coeffs_prev[i - j + 1]

        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp

        q = dtype(1) - reflect_coeff**2
        den = q*den - bwd_pred_error[-1]**2 - fwd_pred_error[0]**2

        fwd_pred_error = fwd_pred_error[1:]
        bwd_pred_error = bwd_pred_error[:-1]

    return ar_coeffs


def lpc_spectrum(lpc_coeffs, f_s, n_fft, return_db=True, return_f_positive=True, normalize=False):
    
    if not return_f_positive:
        f = np.fft.fftfreq(n_fft)*f_s
    else:
        f = np.fft.rfftfreq(n_fft)*f_s
    
    m_order = len(lpc_coeffs)
    lpc_comps_fd = np.array([lpc_coeffs[m]*np.exp(-2*np.pi*1j*m*(f/f_s)) for m in range(m_order)])
    lpc_fd = 1/np.sum(lpc_comps_fd, axis=0)
    
    if normalize:
        normalize = lambda x: x/np.sqrt(np.sum(x**2))
        lpc_fd = normalize(lpc_fd)
    
    if not return_db:
        return lpc_fd
    
    else:
        lpc_fd_db = 20*np.log10(np.abs(lpc_fd))
        return lpc_fd_db


def lpc_frames(wave, order, f_s, n_window, hop_length):
    
    # windowing:

    idx_windows = librosa.time_to_frames(np.arange(len(wave)), sr=f_s, n_fft=n_window, hop_length=hop_length)
    wave_padded = np.pad(wave, int(n_window//2), mode='reflect')
    wave_windows = librosa.util.frame(wave_padded, frame_length=n_window, hop_length=hop_length).T

    window = scipy.signal.windows.hann(n_window)

    # LPC coefficients:

    lpc_coeffs_list = []

    for w in wave_windows:
        lpc_coeffs = lpc(w*window, order=order)
        lpc_coeffs_list.append(lpc_coeffs)

    lpc_coeffs_frames = np.stack(lpc_coeffs_list).T
    
    return lpc_coeffs_frames


def lpc_spectrogram(lpc_coeffs_frames, 
                    f_s, n_fft, 
                    scale='db_scaled', 
                    return_f_positive=True, 
                    return_mel=True, n_mels=256):

    # spectrogram:
    
    if not return_f_positive:
        f = np.fft.fftfreq(n_fft)*f_s
    else:
        f = np.fft.rfftfreq(n_fft)*f_s

    m_order = lpc_coeffs_frames.shape[0]
    lpc_comps_stft = np.array([np.outer(lpc_coeffs_frames[m], np.exp(-2*np.pi*1j*m*(f/f_s))) for m in range(m_order)])
    lpc_stft = 1/np.sum(lpc_comps_stft, axis=0)
    lpc_stft = lpc_stft.T
    
    lpc_spec = {}
    lpc_spec['stft'] = lpc_stft
    
    # mel spectrogram:
    
    if return_mel:
        mel_filters = librosa.filters.mel(sr=f_s, n_fft=n_fft, n_mels=n_mels)
        lpc_mel = np.sqrt(np.dot(mel_filters, np.abs(lpc_stft)**2))
        lpc_spec['mel'] = lpc_mel
    
    # re-scale:
    
    assert scale in ['linear', 'db', 'db_scaled']
    
    for key in lpc_spec:
        if scale == 'db':
            lpc_spec[key] = 20*np.log10(lpc_spec[key])
        elif scale == 'db_scaled':
            lpc_spec[key] = utils.trans.scale_magnitude(np.abs(lpc_spec[key]))
            
    return lpc_spec

