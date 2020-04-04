# imports:

import utils.extract

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
import matplotlib

# config:

figsize = (15, 2)

cmap_spec = 'magma'
cmap_phase = plt.cm.rainbow

color_dict  = {
    'red':  ((0, 0, 0), (1, 0, 0)),
    'green': ((0, 0, 0), (1, 0, 0)),
    'blue':  ((0, 0, 0), (1, 0, 0)),
    'alpha':  ((0, 1, 1), (1, 0, 0))
}

cmap_magnitude = matplotlib.colors.LinearSegmentedColormap('Mask', color_dict)

idx_cut_cc = 2


# functions:

def display_shapes(x):
    
    print('x' + ':')
    for out_key in x.keys():
        if type(x[out_key])==dict:
            print('--' + out_key + ':')
            for in_key in x[out_key].keys():
                assert type(x[out_key][in_key])==np.ndarray
                print('  --' + in_key  + ': ' + str(x[out_key][in_key].shape))
        elif type(x[out_key])==np.ndarray:
            print('--' + out_key  + ': ' + str(x[out_key].shape))
            
    return


def display_all(
    x, 
    figsize=figsize, 
    cmap_spec=cmap_spec, 
    display_rainbowgrams=False, 
    display_cepstrum=False):
    
    # create axes:
    axes = create_ticks_all(x)

    # sound wave:
    display(Audio(x['wave']['s'], rate=utils.extract.f_s))

    # display wave:
    plt.figure(figsize=figsize)
    plt.plot(x['wave']['t'], x['wave']['s'])
    plt.xlim(x['wave']['t'][[0, -1]])
    plt.xticks(axes['wave']['t']['val'], axes['wave']['t']['val'])
    plt.xlabel('t [sec]')
    plt.title('Waveform')
    plt.grid()
    plt.show()

    # display stft:
    plt.figure(figsize=figsize)
    plt.imshow(x['stft']['s'], origin='lower', aspect='auto', cmap=cmap_spec)
    plt.xticks(axes['stft']['t']['idx'], axes['stft']['t']['val'])
    plt.yticks(axes['stft']['f']['idx'], axes['stft']['f']['val'])
    plt.xlabel('t [sec]')
    plt.ylabel('f [KHz]')
    plt.title('Short-Time Fourier Transform')
    plt.show()
    
    # display fft:
    plt.figure(figsize=figsize)
    plt.plot(x['fft']['f'], x['fft']['s'])
    plt.xlim(x['fft']['f'][[0, -1]])
    plt.xticks(axes['fft']['f']['val'], axes['fft']['f']['val'])
    plt.xlabel('f [Hz]')
    plt.title('Spectrum')
    plt.grid()
    plt.show()
    
#     # display stft (with phase):
#     if display_rainbowgrams:
#         plt.figure(figsize=figsize)
#         plt.imshow(x['stft']['f_phase'], origin='lower', aspect='auto', cmap=cmap_phase)
#         plt.imshow(x['stft']['s'], origin='lower', aspect='auto', cmap=cmap_magnitude)
#         plt.xticks(axes['stft']['t']['idx'], axes['stft']['t']['val'])
#         plt.yticks(axes['stft']['f']['idx'], axes['stft']['f']['val'])
#         plt.xlabel('t [sec]')
#         plt.ylabel('f [KHz]')
#         plt.title('Short-Time Fourier Transform Rainbowgram')
#         plt.show()
        
#     # display stft cepstrum:
#     if display_cepstrum:
#         plt.figure(figsize=figsize)
#         plt.imshow(x['stft']['c'][idx_cut_cc:], origin='lower', aspect='auto', cmap=cmap_spec)
#         plt.xticks(axes['stft']['t']['idx'], axes['stft']['t']['val'])
#         plt.xlabel('t [sec]')
#         plt.ylabel('coefficients')
#         plt.title('Short-Time Fourier Cepstral Coefficients')
#         plt.ylim([0, None]) ###
#         plt.show()

#     # display mel:
#     plt.figure(figsize=figsize)
#     plt.imshow(x['mel']['s'], origin='lower', aspect='auto', cmap=cmap_spec)
#     plt.xticks(axes['mel']['t']['idx'], axes['mel']['t']['val'])
#     plt.yticks(axes['mel']['f']['idx'], axes['mel']['f']['val'])
#     plt.xlabel('t [sec]')
#     plt.ylabel('f [KHz]')
#     plt.title('Mel-Frequency Transform')
#     plt.show()
    
#     # display mel cepstrum (mfcc):
#     if display_cepstrum:
#         plt.figure(figsize=figsize)
#         plt.imshow(x['mel']['c'][idx_cut_cc:], origin='lower', aspect='auto', cmap=cmap_spec)
#         plt.xticks(axes['mel']['t']['idx'], axes['mel']['t']['val'])
#         plt.xlabel('t [sec]')
#         plt.ylabel('coefficients')
#         plt.title('Mel-Frequency Cepstral Coefficients')
#         plt.ylim([0, 60]) ###
#         plt.show()

#     # display cqt:
#     plt.figure(figsize=figsize)
#     plt.imshow(x['cqt']['s'], origin='lower', aspect='auto', cmap=cmap_spec)
#     plt.xticks(axes['cqt']['t']['idx'], axes['cqt']['t']['val'])
#     plt.yticks(axes['cqt']['f']['idx'], axes['cqt']['f']['val'])
#     plt.xlabel('t [sec]')
#     plt.ylabel('f [KHz]')
#     plt.title('Constant-Q Transform')
#     plt.show()
    
#     # display cqt (with phase):
#     if display_rainbowgrams:
#         plt.figure(figsize=figsize)
#         plt.imshow(x['cqt']['f_phase'], origin='lower', aspect='auto', cmap=cmap_phase)
#         plt.imshow(x['cqt']['s'], origin='lower', aspect='auto', cmap=cmap_magnitude)
#         plt.xticks(axes['cqt']['t']['idx'], axes['cqt']['t']['val'])
#         plt.yticks(axes['cqt']['f']['idx'], axes['cqt']['f']['val'])
#         plt.xlabel('t [sec]')
#         plt.ylabel('f [KHz]')
#         plt.title('Constant-Q Transform Rainbowgram')
#         plt.show()
        
#     # display mel cepstrum (cqcc):
#     if display_cepstrum:
#         plt.figure(figsize=figsize)
#         plt.imshow(x['cqt']['c'][idx_cut_cc:], origin='lower', aspect='auto', cmap=cmap_spec)
#         plt.xticks(axes['cqt']['t']['idx'], axes['cqt']['t']['val'])
#         plt.xlabel('t [sec]')
#         plt.ylabel('coefficients')
#         plt.title('Constant-Q Cepstral Coefficients')
#         plt.ylim([0, 60]) ###
#         plt.show()
        
#     # display features:
#     plt.figure(figsize=figsize)
#     idx = ~np.isnan(x['features'])
#     val = np.std(x['features'][idx])
#     plt.imshow(x['features'], aspect='auto', origin='lower', cmap=cmap_spec, vmin=-val, vmax=val)
#     plt.xticks(axes['stft']['t']['idx'], axes['stft']['t']['val'])
#     plt.xlabel('t [sec]')
#     plt.ylabel('features')
#     plt.title('Extracted Features')
#     plt.show()
    
    return


def create_ticks(val_axis, n_ticks=9, n_decimals=2):

    idx_ticks = np.linspace(0, len(val_axis)-1, n_ticks).astype(int)
    val_ticks = val_axis[idx_ticks]
    val_ticks = np.round(val_ticks, n_decimals)

    return idx_ticks, val_ticks


def create_ticks_all(x):
    
    axes = {}

    for out_key in x.keys():
        axes[out_key] = {}
        if type(x[out_key])==dict:
            for in_key in x[out_key].keys():
                if in_key in ['f', 't']:
                    idx_ticks, val_ticks = create_ticks(
                        x[out_key][in_key], n_ticks=9, n_decimals=2)
                    axes[out_key][in_key] = {
                        'idx': idx_ticks,
                        'val': val_ticks}
    return axes