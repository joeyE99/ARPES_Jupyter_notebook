import numpy as np
import h5py as h5py

########################################################################################################################

def import_h5py(filepath):
    with h5py.File(filepath, 'r') as f:
        # 1. Access the dataset object (does not load data into memory yet)
        ds = f['Electron Analyzer']['Image Data']
        attrs = ds.attrs
    
        # 2. Extract scales using a helper function to avoid repetition
        def get_axis(axis_idx, length):
            start, step = attrs[f'Axis{axis_idx}.Scale']
            return np.arange(start, start + step * (length - 0.5), step)

        # Calculate lengths based on the raw dataset shape (typically [Z, Y, X] or [Y, X])
        raw_shape = ds.shape
    
        data0 = ds[:]
        axes = [get_axis(i, dim_len) for i, dim_len in enumerate(raw_shape)]
    
        # Metadata
        E_work = attrs['Work Function (eV)']
        T = f['Other Instruments']['Temperature B (Sample 1)'][0]
        hv = np.array([attrs['Excitation Energy (eV)']])

        return data0, axes, hv, E_work, T
