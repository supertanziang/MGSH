import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
import numpy as np
import h5py
from tqdm import tqdm
import pdb
import scipy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np

def is_hdf5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            return True
    except OSError:
        return False


def load_mat(file_path):
    if is_hdf5(file_path):
        print("This is an HDF5 (v7.3) format .mat file.")
        file = h5py.File(file_path, 'r')
        return file, True
    else:
        print("This is an older (v7) format .mat file.")
        data = scipy.io.loadmat(file_path)
        return data, False

def generate_meta_dataset(
    input_h5_path: str,
    output_h5_path: str,
    split_name: str = 'dataBase',
    num_meta: int = None,
    meta_ratio: float = None,
    seed: int = 0
):
    """
    Extract samples from the HDF5 file at input_h5_path using split_name (e.g., 'database').
    You can specify a fixed number (num_meta) or a ratio (meta_ratio, e.g., 0.01 for 1% of all samples).
    meta_ratio takes precedence if both are provided.
    Write the results to output_h5_path with keys: 'ImgMeta', 'LabMeta', and 'TagMeta'.
    """

    import os
    import numpy as np
    import h5py

    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

    with h5py.File(input_h5_path, 'r') as f_in:
        imgs_key = f'Img{split_name}'
        labs_key = f'Lab{split_name}'
        tags_key = f'Tag{split_name}'
        imgs_all = f_in[imgs_key][:]
        labs_all = f_in[labs_key][:]
        tags_all = f_in[tags_key][:]

    total = imgs_all.shape[0]

    # If a ratio is specified, calculate the number based on the ratio.
    if meta_ratio is not None:
        num_meta = max(1, int(total * meta_ratio))
    elif num_meta is None:
        raise ValueError("Either num_meta or meta_ratio must be specified.")

    if num_meta > total:
        raise ValueError(f"Requested to extract {num_meta} samples, but only {total} samples are available.")

    np.random.seed(seed)
    indices = np.random.choice(total, size=num_meta, replace=False)

    imgs_meta = imgs_all[indices]
    labs_meta = labs_all[indices]
    tags_meta = tags_all[indices]

    with h5py.File(output_h5_path, 'w') as f_out:
        f_out.create_dataset('ImgMeta', data=imgs_meta, compression='gzip')
        f_out.create_dataset('LabMeta', data=labs_meta, compression='gzip')
        f_out.create_dataset('TagMeta', data=tags_meta, compression='gzip')

    print(f"Successfully created meta dataset: {output_h5_path}, with {num_meta} samples, accounting for {num_meta/total:.2%} of the original data.")




if __name__ == "__main__":  
    META_RATE = 0.02
    file_path = r'data/nus-wide-tc10-xall-vgg-clean.mat'
    data, is_hdf5_format = load_mat(file_path)
    # plot_and_output_row(data, is_hdf5_format, row_idx=0)  
    generate_meta_dataset(
        input_h5_path  = 'data/NUS-WIDE.h5',
        output_h5_path = 'meta/NUS-WIDE-meta.h5',
        split_name     = 'DataBase',
        # num_meta =  2000,
        meta_ratio       = META_RATE,
        seed           = 0
    )