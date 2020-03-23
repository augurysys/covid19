import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm_notebook as tqdm


def load_wave(file_path):

    x_wave, f_s = librosa.load(file_path, sr=None, mono=True)
    t = np.arange(len(x_wave))/f_s
    
    x_dict = {
        's': x_wave,
        't': t
    }
    
    return x_dict


def load_wave_list(dir_path):

    src_path = Path(dir_path)
    fnames = [f.name for f in src_path.iterdir()]

    x_dict_list = []

    with tqdm(total=len(fnames), unit='files') as pbar:
        for file_name in fnames:
            file_path = dir_path + '\\' + file_name
            x_dict = load_wave(file_path)
            pbar.update()
            
            x_dict_list.append(x_dict)
    
    return x_dict_list