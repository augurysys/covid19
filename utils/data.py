# imports:

import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm_notebook as tqdm
import pandas as pd


# functions:

def load_wave(file_path):

    x_wave, f_s = librosa.load(file_path, sr=None, mono=True)
    t = np.arange(len(x_wave))/f_s
    
    x_dict = {
        's': x_wave,
        't': t
    }
    
    return x_dict


def load_wave_files(file_path_str_list):

    x_wave_list = []
    f_s_list = []

    with tqdm(total=len(file_path_str_list), unit='file') as pbar:
        for file_path in file_path_str_list:
            x_wave, f_s = librosa.load(file_path, sr=None)
            
            x_wave_list.append(x_wave)
            f_s_list.append(f_s)
            pbar.update()
    
    return x_wave_list, f_s_list


def scan_dir(dir_path):
    
    init_path_list = []
    file_path_list = scan_dir_recursive(dir_path, init_path_list)
    print('%d files were found' %len(file_path_list))
    
    return file_path_list


def scan_dir_recursive(dir_path, file_path_list):
    
    for child in dir_path.iterdir():
        if not child.is_dir():
            file_path_list.append(child)
        else:
            scan_dir_recursive(child, file_path_list)
            
    return file_path_list


def files_dataframe(dir_path_str):

    dir_path = Path(dir_path_str)
    file_path_list = scan_dir(dir_path)

    file_path_str_list = [str(f) for f in file_path_list]
    file_name_list = [f.name for f in file_path_list]
    file_type_list = [f.suffix for f in file_path_list]
    file_dir_list = [f.parent.name for f in file_path_list]

    dict_files = {
        'path': file_path_str_list,
        'name': file_name_list,
        'type': file_type_list,
        'dir': file_dir_list,

    }

    df_files = pd.DataFrame(dict_files)
    
    return df_files