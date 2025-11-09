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


def add_noise_to_labels(labels, noise_rate):
    """
    Add noise to a multi-label one-hot matrix:
        - Randomly select a portion of samples (based on noise_rate)
        - For each selected sample:
            - Flip one positive label (1 → 0)
            - Flip one negative label (0 → 1)
    Args:
        labels: np.ndarray, shape (N, C), original multi-label one-hot matrix
        noise_rate: float in [0, 1], proportion of samples to inject noise into
    Returns:
        labels: np.ndarray, shape (N, C), label matrix with noise
    """
    
    num_samples, num_labels = labels.shape
    num_noise = int(num_samples * noise_rate)
    noise_indices = np.random.choice(num_samples, num_noise, replace=False)

    for i in tqdm(noise_indices):
        ones_indices = np.where(labels[i, :] == 1)[0]
        zeros_indices = np.where(labels[i, :] == 0)[0]
        if len(ones_indices) > 0:
            j1 = np.random.choice(ones_indices)
            labels[i, j1] = 0
        if len(zeros_indices) > 0:
            j0 = np.random.choice(zeros_indices)
            labels[i, j0] = 1
    return labels


def generate_noise_F(noise):
    noise_rate = noise
    data = h5py.File('./data/MIRFlickr.h5', 'r')
    for i in noise_rate:
        labels_matrix = np.array(list(data['LabTrain']))
        labels_matrix2 = np.array(list(data['LabTrain']))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)

        output_file = h5py.File('./noise/mirflickr25k-lall-noise_{}.h5'.format(i), 'w')

        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)

        output_file.close()

def generate_noise_N(noise):
    noise_rate = noise
    data = h5py.File('./data/NUS-WIDE.h5', 'r')
    for i in noise_rate:

        labels_matrix = np.array(list(data['LabTrain']))
        labels_matrix2 = np.array(list(data['LabTrain']))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)

        output_file = h5py.File('./noise/nus-wide-tc10-lall-noise_{}.h5'.format(i), 'w')

        output_file.create_dataset('noisy', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)

        output_file.close()

def generate_noise_M(noise_rates, 
                     h5_path='/data1/tza/NRCH-master/data/MS-COCO_rand_combined.h5', 
                     out_dir='./noise'):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Read original labels
    with h5py.File(h5_path, 'r') as hf:
        print("原 H5 键名:", list(hf.keys()))
        if 'LabTrain' not in hf:
            raise KeyError("Cannot find 'LabTrain', please check the key name.")
        labels = hf['LabTrain'][:]  

    N, C = labels.shape
    for rate in noise_rates:
        # 2) Make a deep copy for noise injection.
        noisy_labels = labels.copy()

        # 3) Randomly select rate * N sample indices at the sample level.
        num_noisy = int(round(N * rate))
        noisy_idx = np.random.choice(N, num_noisy, replace=False)

        # 4) Add noise to the selected samples one by one.
        for i in noisy_idx:
            ones = np.where(noisy_labels[i] == 1)[0]
            zeros = np.where(noisy_labels[i] == 0)[0]
            if len(ones) > 0:
                j1 = np.random.choice(ones)
                noisy_labels[i, j1] = 0
            if len(zeros) > 0:
                j0 = np.random.choice(zeros)
                noisy_labels[i, j0] = 1

        # 5) Write to H5: store sample indices in 'noisy', and store label matrices in other datasets.
        out_path = os.path.join(out_dir, f"ms-coco-lall-noise_{rate:.1f}.h5")
        with h5py.File(out_path, 'w') as hf_out:
            hf_out.create_dataset('noisy',        data=np.sort(noisy_idx), dtype=noisy_idx.dtype)
            hf_out.create_dataset('noisy_labels', data=noisy_labels,        dtype=labels.dtype)
            hf_out.create_dataset('true_labels',  data=labels,               dtype=labels.dtype)

        print(f"[OK] Noise rate {rate:.1f}，Number of noisy samples {len(noisy_idx)}/{N} → {out_path}")

def generate_noise_I(noise):
    noise_rate = noise
    data = h5py.File('./data/IAPR.h5', 'r')
    for i in noise_rate:
        labels_matrix = np.array(list(data['LabTrain']))
        labels_matrix2 = np.array(list(data['LabTrain']))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)

        output_file = h5py.File('./noise/IAPR-lall-noise_{}.h5'.format(i), 'w')

        output_file.create_dataset('noisy', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)
        
        output_file.close()

def is_hdf5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            return True
    except OSError:
        return False
   
def plot_and_output_row(data, is_hdf5_format, row_idx=0):
    if is_hdf5_format:
        keys = list(data.keys())
        print(f"Variable list (HDF5): {keys}")
        var_name = keys[1]
        dataset = data[var_name][()]
        dataset = np.array(dataset)  
    else:
        keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Variable list: {keys}")
        var_name = keys[0]
        dataset = data[var_name]

    print(f"Shape of variable {var_name}: {dataset.shape}")

    if dataset.ndim >= 2:
        if row_idx < dataset.shape[0]:
            print(f"Values of row {row_idx}:")
            print(dataset[row_idx])
        else:
            print(f"Row index {row_idx} is out of range; total number of rows is {dataset.shape[0]}.")
    elif dataset.ndim == 1:
        print(f"This is 1D data, full content:")
        print(dataset)
    else:
        print("Data dimensionality is too high; direct row output is not supported.")

    # Plotting
    if dataset.ndim == 1:
        plt.plot(dataset)
        plt.title(f'{var_name} (1D Plot)')
        plt.xlabel('Index')
        plt.ylabel('Value')
    elif dataset.ndim == 2:
        plt.imshow(dataset, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'{var_name} (2D Heatmap)')
    elif dataset.ndim == 3:
        slice_idx = dataset.shape[2] // 2
        plt.imshow(dataset[:, :, slice_idx], cmap='gray')
        plt.title(f'{var_name} (Slice {slice_idx})')
    else:
        print("Data dimensionality is too high or unrecognized; plotting is not supported.")

    plt.show()
  
if __name__ == "__main__":  
    noise_rate = [0.2,0.5,0.8]
    generate_noise_N(noise_rate)

