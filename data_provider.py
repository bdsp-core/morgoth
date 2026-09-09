import os
import mne
from flatbuffers.compat import binary_types
from numba.cuda.tests.nocuda.test_nvvm import original

mne.set_log_level(verbose='WARNING')
from itertools import chain
import numpy as np
import random
import h5py
import hdf5storage
import bisect
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
from multiprocessing import Pool
import pickle
from tqdm import tqdm
import glob
import pandas as pd
from datetime import timedelta,datetime,timezone
import pytz
from sklearn.preprocessing import MinMaxScaler
import shutil
import json
import scipy.io
import mat73
import sys
from scipy.signal import resample
import re
import ast
from scipy.signal import butter, filtfilt, iirnotch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
list_path = List[Path]


class ringBuffer:
    def __init__(self, buffer:list, shuffle:bool=False):
        if shuffle:
            random.shuffle(buffer)
        self.__buffer = buffer
        self.__maxSize = len(buffer)
        self.__head = 0

    def sample(self, num:int) -> list:
        sampleList = []
        for idx in range(num):
            sampleList.append(self.__buffer[self.__head])
            self.__head = (self.__head + 1) % self.__maxSize
        return sampleList

    @property
    def data(self):
        return self.__buffer



class SingleShockDataset(Dataset):
    """Read single hdf5 file regardless of label, subject, and paradigm."""

    def __init__(self, file_path: Path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0,
                 end_percentage: float = 1):
        '''
        Extract datasets from file_path.

        param Path file_path: the path of target data
        param int window_size: the length of a single sample
        param int stride_size: the interval between two adjacent samples
        param float start_percentage: Index of percentage of the first sample of the dataset in the data file (inclusive)
        param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
        '''
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__global_idxes = []
        self.__local_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        for subject in self.__subjects:
            self.__global_idxes.append(global_idx)
            subject_len = self.__file[subject]['eeg'].shape[1]
            total_sample_num = (subject_len - self.__window_size) // self.__stride_size + 1
            start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size
            end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size

            self.__local_idxes.append(start_idx)
            global_idx += (end_idx - start_idx) // self.__stride_size + 1
        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        subject_idx = bisect.bisect(self.__global_idxes, idx) - 1
        item_start_idx = (idx - self.__global_idxes[subject_idx]) * self.__stride_size + self.__local_idxes[subject_idx]
        return self.__file[self.__subjects[subject_idx]]['eeg'][:, item_start_idx:item_start_idx + self.__window_size]

    def free(self) -> None:
        if self.__file:
            self.__file.close()
            self.__file = None

    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


class ShockDataset(Dataset):
    """integrate multiple hdf5 files"""

    def __init__(self, file_paths: list_path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0,
                 end_percentage: float = 1):
        '''
        Arguments will be passed to SingleShockDataset. Refer to SingleShockDataset.
        '''
        self.__file_paths = file_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__datasets = []
        self.__length = None
        self.__feature_size = None

        self.__dataset_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [
            SingleShockDataset(file_path, self.__window_size, self.__stride_size, self.__start_percentage,
                               self.__end_percentage) for file_path in self.__file_paths]

        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx])
        return self.__datasets[dataset_idx][item_idx]

    def free(self) -> None:
        for dataset in self.__datasets:
            dataset.free()

    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()



eeg_channels=['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

eeg_channels_with_EKG=['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']

eeg_channels_avg=['FP1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'F7-AVG', 'T3-AVG', 'T5-AVG', 'O1-AVG', 'FZ-AVG', 'CZ-AVG', 'PZ-AVG', 'FP2-AVG', 'F4-AVG', 'C4-AVG', 'P4-AVG', 'F8-AVG',
                    'T4-AVG', 'T6-AVG', 'O2-AVG']

class h5Dataset:
    def __init__(self, path: Path, name: str, rewrite: bool = True) -> None:
        self.__name = name
        file_path = path / f'{self.__name}.hdf5'

        if file_path.exists() and rewrite:
            os.remove(file_path)

        self.__f = h5py.File(file_path, 'a')

    def addGroup(self, grpName: str):
        return self.__f.create_group(grpName), False

    def addDataset(self, grp: h5py.Group, dsName: str, arr: np.array, chunks: tuple):
        return grp.create_dataset(dsName, data=arr, chunks=chunks)

    def addAttributes(self, src: 'h5py.Dataset|h5py.Group', attrName: str, attrValue):
        src.attrs[f'{attrName}'] = attrValue

    def save(self):
        self.__f.close()

    def list_groups_with_datasets(self):
        """List all group names that contain at least one dataset."""
        groups_with_datasets = []

        def find_datasets(name, obj):
            if isinstance(obj, h5py.Group):
                for item in obj:
                    if isinstance(obj[item], h5py.Dataset):
                        groups_with_datasets.append(name)
                        break

        self.__f.visititems(find_datasets)

        return list(set(groups_with_datasets))

    def delete_group(self, grpName: str):
        """Delete a specified group."""
        if grpName in self.__f:
            del self.__f[grpName]
            print(f"Group '{grpName}' has been deleted.")
        else:
            print(f"Group '{grpName}' does not exist.")

    @property
    def name(self):
        return self.__name




def bipolar(data_monopolar):
    data_bipolar = np.zeros((18, data_monopolar.shape[1]))
    data_bipolar[0, :] = data_monopolar[0, :] - data_monopolar[4, :]
    data_bipolar[1, :] = data_monopolar[4, :] - data_monopolar[5, :]
    data_bipolar[2, :] = data_monopolar[5, :] - data_monopolar[6, :]
    data_bipolar[3, :] = data_monopolar[6, :] - data_monopolar[7, :]

    data_bipolar[4, :] = data_monopolar[11, :] - data_monopolar[15, :]
    data_bipolar[5, :] = data_monopolar[15, :] - data_monopolar[16, :]
    data_bipolar[6, :] = data_monopolar[16, :] - data_monopolar[17, :]
    data_bipolar[7, :] = data_monopolar[17, :] - data_monopolar[18, :]

    data_bipolar[8, :] = data_monopolar[0, :] - data_monopolar[1, :]
    data_bipolar[9, :] = data_monopolar[1, :] - data_monopolar[2, :]
    data_bipolar[10, :] = data_monopolar[2, :] - data_monopolar[3, :]
    data_bipolar[11, :] = data_monopolar[3, :] - data_monopolar[7, :]

    data_bipolar[12, :] = data_monopolar[11, :] - data_monopolar[12, :]
    data_bipolar[13, :] = data_monopolar[12, :] - data_monopolar[13, :]
    data_bipolar[14, :] = data_monopolar[13, :] - data_monopolar[14, :]
    data_bipolar[15, :] = data_monopolar[14, :] - data_monopolar[18, :]

    data_bipolar[16, :] = data_monopolar[8, :] - data_monopolar[9, :]
    data_bipolar[17, :] = data_monopolar[9, :] - data_monopolar[10, :]

    channel_names = ["Fp1-F7","F7-T3","T3-T5","T5-O1","Fp2-F8","F8-T4","T4-T6","T6-O2","Fp1-F3","F3-C3","C3-P3","P3-O1","Fp2-F4","F4-C4","C4-P4","P4-O2","Fz-Cz","Cz-Pz"]


    return data_bipolar,channel_names


def preprocessing_edf(edf_path, l_freq=0.5, h_freq=70.0, sfreq=200):
    eeg_channels1 = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                     'T4', 'T6', 'O2']
    eeg_channels2 = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T7', 'P7', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                     'T8', 'P8', 'O2']

    if not os.path.exists(edf_path):
        print (f"{edf_path} does not exist")
        return None, None

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True)
    except Exception as e:
        print(f"Failed to read {edf_path}: {e}")
        return None, None

    if raw.times[-1] > 3600:
        raw = raw.crop(tmin=0, tmax=3600)

    elif raw.times[-1] < 11:
        print(f"{edf_path} is too short (<= 10 minutes)")
        return None, None

    new_channel_names = {ch_name: ch_name.upper() for ch_name in raw.ch_names}
    raw.rename_channels(new_channel_names)

    channels = raw.ch_names
    if set(channels).issuperset(set(eeg_channels1)):
        selected_channels = eeg_channels1
    elif set(channels).issuperset(set(eeg_channels2)):
        selected_channels = eeg_channels2
    else:
        print(f"{edf_path} does not contain all 19 required channels")
        return None, False

    raw_selected = raw.copy().pick_channels(selected_channels)
    raw_selected = raw_selected.resample(sfreq, n_jobs=5)
    raw_selected = raw_selected.filter(l_freq=l_freq, h_freq=h_freq)
    raw_selected = raw_selected.notch_filter(60.0)
    raw_selected = raw_selected.notch_filter(50.0)
    raw_selected.set_eeg_reference('average')

    eegData = raw_selected.get_data(units='uV')
    eegData=EEG_clip(eegData)

    selected_channel_names = raw_selected.ch_names

    return eegData, selected_channel_names


def processing_mat(file,l_freq, h_freq, rsfreq,used_channels):
    file=str(file)
    raw = hdf5storage.loadmat(file)
    data=raw['data']
    channels = get_channel_names_from_mat(raw)
    Fs=get_frequency_from_mat(raw)
    data_dic = dict(zip(channels, data))
    data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=used_channels,
                                        default_value=np.array([0] * data.shape[1]))
    data = np.array(list(data_dic.values()))

    if np.isnan(data).any():
        data, _, _= remove_nan_columns(data)
        print(f"Raw data contains NaN values: {file}, remove nan and change event center.")

    if np.any(np.all(np.abs(data) > 5000, axis=0)):
        data, _= find_valid_segment(data=data, n=data.shape[1]//2)

    if data.shape[1]<segment_size*Fs:
        print(f"{file} valid length not enough ")
        return None,None

    eegData = resample_signal(signal=data, original_rate=Fs, target_rate=rsfreq)
    eegData=EEG_bandfilter(eegData,fs=rsfreq, order=4, low=l_freq, high=h_freq)
    eegData=EEG_notchfilter(eegData,fs=rsfreq, notch_width=1.0)

    eegData=EEG_avg(eegData)
    eegData=EEG_clip(eegData)

    if np.isnan(data).any():
        print(f"{file} contains NaN values after processing")
        return None,None

    return eegData, eeg_channels



def make_hdf5_for_MGB(savePath='/data/PRETRAIN_DATASET/', dataset_name='dataset', eeg_type='mat',rewrite=True):
    os.makedirs(savePath, exist_ok=True)
    savePath=Path(savePath)

    dataset = h5Dataset(path=savePath, name=dataset_name, rewrite=rewrite)

    l_freq = 0.5
    h_freq = 70.0
    rsfreq = 200

    monopolar_channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

    chunks = (19, rsfreq)

    data_list_files=['list_bets_2024Nov17.xlsx',
                     'list_BS_20250117.xlsx',
                     'list_fs_2024Nov17.xlsx',
                     'list_gpd_2024Nov17.xlsx',
                     'list_grda_2024Nov17.xlsx',
                     'list_gs_2024Nov17.xlsx',
                     'list_lpd_2024Nov17.xlsx',
                     'list_lrda_2024Nov17.xlsx',
                     'list_pdr_2024Nov17.xlsx',
                     'list_posts_2024Nov17.xlsx',
                     'list_seizures_2024Nov17.xlsx',
                     'list_seizures_bch_20241201.xlsx',
                     'list_iiic_20241129.xlsx',
                     'list_spikes_bch_20241126.xlsx',
                     'list_spindles_2024Nov17.xlsx',
                     'list_vw_2024Nov17.xlsx',
                     'list_wickets_2024Nov17.xlsx'
    ]

    bdsp_mrn_dic = {'BETS':set(),
                'BS':set(),
                'FS':set(),
                'GPD':set(),
                'GRDA':set(),
                'GS':set(),
                'LPD':set(),
                'LRDA':set(),
                'PDR':set(),
                'POSTS':set(),
                'SEIZURE':set(),
                'SEIZURE_BCH':set(),
                'IIIC' :set(),
                'SPIKES_BCH':set(),
                'SPINDLES':set(),
                'VW':set(),
                'WICKETS' :set(),
                 }

    print(f'[*] Make pretrain dataset list')
    bdsp_mrn=set()

    for list_file,(feature, list_set) in zip(data_list_files,bdsp_mrn_dic.items()):
        list_file_path = Path(f'/data/Dataset_statistic/{list_file}')
        list_df = pd.read_excel(list_file_path, sheet_name=None)

        for sheet_name, data in list_df.items():
            if 'bdsp_mrn' in data.columns:
                data['bdsp_mrn'] = data['bdsp_mrn'].astype(str)
                new_bdsp_mrn= set(data['bdsp_mrn'].dropna().tolist())
            else:
                print(f"No bdsp_mrn in {list_file}")

        new_bdsp_mrn = new_bdsp_mrn - bdsp_mrn
        selected_num= min(len(new_bdsp_mrn), 500)
        print(f'selected {selected_num} patients for {feature}')
        new_bdsp_mrn = random.sample(list(new_bdsp_mrn),selected_num)

        list_set.update(new_bdsp_mrn)
        bdsp_mrn.update(new_bdsp_mrn)

    bdsp_mrn_dic['NORMAL']=set()
    raw_mat_path=Path(f'/data/NORMAL/segments_raw')
    feature_files = list(raw_mat_path.glob(f'*.{eeg_type}'))

    n_bdsp_mrn=10000-len(bdsp_mrn)
    print(f'selected {n_bdsp_mrn} patients for NORMAL')

    for file in feature_files:
        if n_bdsp_mrn==0: break

        file_stem = file.stem
        key = file_stem.split('_')[0].split('-')[-1][5:]
        if key not in bdsp_mrn:
            bdsp_mrn_dic['NORMAL'].add(key)
            n_bdsp_mrn-=1

    def custom_serializer(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Type {type(obj)} is not JSON serializable")

    with open('/data/PRETRAIN_DATASET_list.json', 'w') as f:
        json.dump(bdsp_mrn_dic, f, indent=4,default=custom_serializer)


    for feature,list_set in bdsp_mrn_dic.items():
        print(f'[*] Make pretrain dataset with {feature}')

        raw_mat_path = Path(f'/data/{feature}/segments_raw')

        if not os.path.exists(raw_mat_path):
            print(f'No {raw_mat_path}')
            continue

        feature_files = list(raw_mat_path.glob(f'*.{eeg_type}'))
        file_key_pairs = []
        for file in feature_files:
            file_stem = file.stem
            key = file_stem.split('_')[0].split('-')[-1][5:]
            if key in list_set:
                file_key_pairs.append((key, file))

        unique_files = {}
        for key, file in file_key_pairs:
            if key not in unique_files:
                unique_files[key] = []
            unique_files[key].append(file)

        feature_files = [random.choice(files) for files in unique_files.values()]
        for file in tqdm(feature_files,desc=f'Make pretrain dataset with {feature}'):
            file = Path(file)

            if eeg_type == 'edf':
                eegData, chOrder = preprocessing_edf(file, l_freq, h_freq, rsfreq)
            elif eeg_type == 'mat':
                eegData, chOrder = processing_mat(file, l_freq, h_freq, rsfreq,used_channels=eeg_channels)
            else:
                print(f"Data format for {file.stem} is invalid. Skipping.")
                continue

            if eegData is None:
                print(f"Data for {file.stem} is invalid. Skipping.")
                continue

            dataset = h5Dataset(path=savePath, name=dataset_name, rewrite=False)

            grp, group_exists = dataset.addGroup(grpName=f'{file.stem}')
            if group_exists:
                print(f"Group {file.stem} already exists. Skipping.")
                continue

            dset = dataset.addDataset(grp, 'eeg', eegData, chunks)
            dataset.addAttributes(dset, 'lFreq', l_freq)
            dataset.addAttributes(dset, 'hFreq', h_freq)
            dataset.addAttributes(dset, 'rsFreq', rsfreq)
            dataset.addAttributes(dset, 'chOrder', chOrder)

    dataset.save()

    print('[*] hdf5 file created.')



def add_hdf5_for_MGB(add_data_path, num_files_to_select=500,savePath='/data/PRETRAIN_DATASET/', dataset_name='dataset',eeg_type='mat'):
    savePath = Path(savePath)
    l_freq = 0.5
    h_freq = 70.0
    rsfreq = 200

    chunks = (19, rsfreq)
    dataset = h5Dataset(path=savePath, name=dataset_name, rewrite=False)

    add_data_path=Path(add_data_path)
    feature_files = list(add_data_path.glob(f'*.{eeg_type}'))
    feature_files = random.sample(feature_files, min(num_files_to_select, len(feature_files)))

    for file in tqdm(feature_files,desc=f'Add pretrain dataset'):
        file = Path(file)

        if eeg_type == 'edf':
            eegData, chOrder = preprocessing_edf(file, l_freq, h_freq, rsfreq)
        elif eeg_type == 'mat':
            eegData, chOrder = processing_mat(file, l_freq, h_freq, rsfreq,used_channels=eeg_channels)
        else:
            print(f"Data format for {file.stem} is invalid. Skipping.")
            continue

        if eegData is None:
            print(f"Data for {file.stem} is invalid. Skipping.")
            continue

        grp, group_exists = dataset.addGroup(grpName=f'{file.stem}')
        if group_exists:
            print(f"Group {file.stem} already exists. Skipping.")
            continue

        dset = dataset.addDataset(grp, 'eeg', eegData, chunks)
        dataset.addAttributes(dset, 'lFreq', l_freq)
        dataset.addAttributes(dset, 'hFreq', h_freq)
        dataset.addAttributes(dset, 'rsFreq', rsfreq)
        dataset.addAttributes(dset, 'chOrder', chOrder)
    
    dataset.save()
    print('[*] added data to hdf5 file.')




def explore_hdf5(file, indent=0):
    for key in file:
        print(' ' * indent + f"Key: {key}")
        if isinstance(file[key], h5py.Group):
            explore_hdf5(file[key], indent + 4)
        else:
            print(' ' * (indent + 4) + str(file[key]))


def check_HDF5_eeg_key(hdf5_file_path='/data/PRETRAIN_DATASET/dataset.hdf5'):
    file = h5py.File(hdf5_file_path, 'r')
    keys_without_eeg = []

    for key in file:
        try:
            eeg_shape = file[key]['eeg'].shape[1]
        except KeyError:
            print(f"Key: {key} does not have 'eeg' dataset")
            keys_without_eeg.append(key)
        except AttributeError:
            print(f"Key: {key} is not a group, skipping...")
            continue
    file.close()
    return keys_without_eeg



drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

standard_channels = ["EEG FP1-REF","EEG F7-REF", "EEG T3-REF", "EEG T5-REF",  "EEG O1-REF", "EEG FP2-REF","EEG F8-REF", "EEG T4-REF", "EEG T6-REF", "EEG O2-REF", "EEG FP1-REF", "EEG F3-REF","EEG C3-REF", "EEG P3-REF","EEG O1-REF","EEG FP2-REF","EEG F4-REF","EEG C4-REF","EEG P4-REF","EEG O2-REF",]



def split_and_dump(params):
    fetch_folder, sub, dump_folder, label = params
    for file in tqdm(os.listdir(fetch_folder)):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            try:
                if drop_channels is not None:
                    useless_chs = []
                    for ch in drop_channels:
                        if ch in raw.ch_names:
                            useless_chs.append(ch)
                    raw.drop_channels(useless_chs)
                if chOrder_standard is not None and len(chOrder_standard) == len(raw.ch_names):
                    raw.reorder_channels(chOrder_standard)
                if raw.ch_names != chOrder_standard:
                    raise Exception("channel order is wrong!")

                raw.filter(l_freq=0.5, h_freq=70.0)
                raw.notch_filter(50.0)
                raw.resample(200, n_jobs=5)

                ch_name = raw.ch_names
                raw_data = raw.get_data(units='uV')
                channeled_data = raw_data.copy()
            except:
                with open("EEGfounder/tuab-process-error-files.txt", "a") as f:
                    f.write(file + "\n")
                continue
            for i in range(channeled_data.shape[1] // 2000):
                dump_path = os.path.join(
                    dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
                )
                pickle.dump(
                    {"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": label},
                    open(dump_path, "wb"),
                )

def TUAB():
    root = "/data/tuh_eeg/TUAB/edf/"
    channel_std = "01_tcp_ar"

    train_val_abnormal = os.path.join(root, "train", "abnormal", channel_std)
    train_val_a_sub = list(
        set([item.split("_")[0] for item in os.listdir(train_val_abnormal)])
    )
    np.random.shuffle(train_val_a_sub)
    train_a_sub, val_a_sub = (
        train_val_a_sub[: int(len(train_val_a_sub) * 0.8)],
        train_val_a_sub[int(len(train_val_a_sub) * 0.8):],
    )

    train_val_normal = os.path.join(root, "train", "normal", channel_std)
    train_val_n_sub = list(
        set([item.split("_")[0] for item in os.listdir(train_val_normal)])
    )
    np.random.shuffle(train_val_n_sub)
    train_n_sub, val_n_sub = (
        train_val_n_sub[: int(len(train_val_n_sub) * 0.8)],
        train_val_n_sub[int(len(train_val_n_sub) * 0.8):],
    )

    test_abnormal = os.path.join(root, "eval", "abnormal", channel_std)
    test_a_sub = list(set([item.split("_")[0] for item in os.listdir(test_abnormal)]))

    test_normal = os.path.join(root, "eval", "normal", channel_std)
    test_n_sub = list(set([item.split("_")[0] for item in os.listdir(test_normal)]))

    if not os.path.exists(os.path.join(root, "processed")):
        os.makedirs(os.path.join(root, "processed"))

    if not os.path.exists(os.path.join(root, "processed", "train")):
        os.makedirs(os.path.join(root, "processed", "train"))
    train_dump_folder = os.path.join(root, "processed", "train")

    if not os.path.exists(os.path.join(root, "processed", "val")):
        os.makedirs(os.path.join(root, "processed", "val"))
    val_dump_folder = os.path.join(root, "processed", "val")

    if not os.path.exists(os.path.join(root, "processed", "test")):
        os.makedirs(os.path.join(root, "processed", "test"))
    test_dump_folder = os.path.join(root, "processed", "test")

    parameters = []
    for train_sub in train_a_sub:
        parameters.append([train_val_abnormal, train_sub, train_dump_folder, 1])
    for train_sub in train_n_sub:
        parameters.append([train_val_normal, train_sub, train_dump_folder, 0])
    for val_sub in val_a_sub:
        parameters.append([train_val_abnormal, val_sub, val_dump_folder, 1])
    for val_sub in val_n_sub:
        parameters.append([train_val_normal, val_sub, val_dump_folder, 0])
    for test_sub in test_a_sub:
        parameters.append([test_abnormal, test_sub, test_dump_folder, 1])
    for test_sub in test_n_sub:
        parameters.append([test_normal, test_sub, test_dump_folder, 0])

    with Pool(processes=24) as pool:
        result = pool.map(split_and_dump, parameters)







def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape
    fs = 200.0
    [numChan, numPoints] = signals.shape
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):
        chan = int(EventData[i, 0])
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels]


def convert_signals(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    new_signals = np.vstack(
        (signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F7-REF"]],
            (signals[signal_names["EEG F7-REF"]]- signals[signal_names["EEG T3-REF"]]),
            (signals[signal_names["EEG T3-REF"]] - signals[signal_names["EEG T5-REF"]]),
            (signals[signal_names["EEG T5-REF"]]- signals[signal_names["EEG O1-REF"]]),
            (signals[signal_names["EEG FP2-REF"]]- signals[signal_names["EEG F8-REF"]]),
            (signals[signal_names["EEG F8-REF"]]- signals[signal_names["EEG T4-REF"]]),
            (signals[signal_names["EEG T4-REF"]]- signals[signal_names["EEG T6-REF"]]),
            (signals[signal_names["EEG T6-REF"]]- signals[signal_names["EEG O2-REF"]]),
            (signals[signal_names["EEG FP1-REF"]]- signals[signal_names["EEG F3-REF"]]),
            (signals[signal_names["EEG F3-REF"]]- signals[signal_names["EEG C3-REF"]]),
            (signals[signal_names["EEG C3-REF"]]- signals[signal_names["EEG P3-REF"]]),
            (signals[signal_names["EEG P3-REF"]]- signals[signal_names["EEG O1-REF"]]),
            (signals[signal_names["EEG FP2-REF"]]- signals[signal_names["EEG F4-REF"]]),
            (signals[signal_names["EEG F4-REF"]]- signals[signal_names["EEG C4-REF"]]),
            (signals[signal_names["EEG C4-REF"]]- signals[signal_names["EEG P4-REF"]]),
            (signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]]),
         )
    )
    return new_signals


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in Rawdata.ch_names:
                useless_chs.append(ch)
        Rawdata.drop_channels(useless_chs)
    if chOrder_standard is not None and len(chOrder_standard) == len(Rawdata.ch_names):
        Rawdata.reorder_channels(chOrder_standard)
    if Rawdata.ch_names != chOrder_standard:
        raise ValueError

    Rawdata.filter(l_freq=0.5, h_freq=70.0)
    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]


def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir):
    for dirName, subdirList, fileList in os.walk(BaseDir):
        print("Found directory: %s" % dirName)
        for fname in tqdm(fileList):
            if fname[-4:] == ".edf":
                try:
                    [signals, times, event, Rawdata] = readEDF(
                        dirName + "/" + fname
                    )
                except (ValueError, KeyError):
                    print("something funky happened in " + dirName + "/" + fname)
                    continue
                signals, offending_channels, labels = BuildEvents(signals, times, event)

                for idx, (signal, offending_channel, label) in enumerate(
                    zip(signals, offending_channels, labels)
                ):
                    sample = {
                        "signal": signal,
                        "offending_channel": offending_channel,
                        "label": label,
                    }
                    save_pickle(
                        sample,
                        os.path.join(
                            OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"
                        ),
                    )

    return Features, Labels, OffendingChannels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)



def TUEV():
    root = "/data/tuh_eeg/TUEV/edf"
    train_out_dir = os.path.join(root, "processed/train")
    eval_out_dir = os.path.join(root, "processed/eval")
    if not os.path.exists(train_out_dir):
        os.makedirs(train_out_dir)
    if not os.path.exists(eval_out_dir):
        os.makedirs(eval_out_dir)

    BaseDirTrain = os.path.join(root, "train")
    fs = 200
    TrainFeatures = np.empty(
        (0, 23, fs)
    )
    TrainLabels = np.empty([0, 1])
    TrainOffendingChannel = np.empty([0, 1])
    load_up_objects(
        BaseDirTrain, TrainFeatures, TrainLabels, TrainOffendingChannel, train_out_dir
    )

    BaseDirEval = os.path.join(root, "eval")
    fs = 200
    EvalFeatures = np.empty(
        (0, 23, fs)
    )
    EvalLabels = np.empty([0, 1])
    EvalOffendingChannel = np.empty([0, 1])
    load_up_objects(
        BaseDirEval, EvalFeatures, EvalLabels, EvalOffendingChannel, eval_out_dir
    )

    root = "/data/tuh_eeg/TUEV/edf"
    seed = 4523
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed/train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.2), replace=False)
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]

    train_sub = list(set(train_sub) - set(val_sub))
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    os.system(f"mv {os.path.join(root, 'processed/eval')} {os.path.join(root, 'processed/test')}")
    print('Done for test')

    os.makedirs(os.path.join(root, 'processed/val/'), exist_ok=True)
    for file in val_files:
        os.system(f"cp {os.path.join(root, 'processed/train/', file)} {os.path.join(root, 'processed/val/',file)}")
        os.system(f"rm {os.path.join(root, 'processed/train/', file)}")
    print('Done for val')
    print('Done for train')




def get_all_files(directory):
    all_files = glob.glob(directory + '/**/*', recursive=True)
    return [file for file in all_files if os.path.isfile(file)]

def TUEP():
    root = "/data/tuh_eeg/TUEP/"

    epilepsy_dir = os.path.join(root, "00_epilepsy")
    epilepsy_sub = list(set(get_all_files(epilepsy_dir)))
    np.random.shuffle(epilepsy_sub)

    train_e_sub, val_e_sub, test_e_sub = (
        epilepsy_sub[: int(len(epilepsy_sub) * 0.6)],
        epilepsy_sub[int(len(epilepsy_sub) * 0.6):int(len(epilepsy_sub) * 0.8)],
        epilepsy_sub[int(len(epilepsy_sub) * 0.8):],
    )

    noepilepsy_dir = os.path.join(root, "01_no_epilepsy")
    noepilepsy_sub = list(set(get_all_files(noepilepsy_dir)))
    np.random.shuffle(noepilepsy_sub)

    train_n_sub, val_n_sub, test_n_sub = (
        noepilepsy_sub[: int(len(noepilepsy_sub) * 0.6)],
        noepilepsy_sub[int(len(noepilepsy_sub) * 0.6):int(len(noepilepsy_sub) * 0.8)],
        noepilepsy_sub[int(len(noepilepsy_sub) * 0.8):],
    )

    if not os.path.exists(os.path.join(root, "processed")):
        os.makedirs(os.path.join(root, "processed"))

    if not os.path.exists(os.path.join(root, "processed", "train")):
        os.makedirs(os.path.join(root, "processed", "train"))
    train_dump_folder = os.path.join(root, "processed", "train")

    if not os.path.exists(os.path.join(root, "processed", "val")):
        os.makedirs(os.path.join(root, "processed", "val"))
    val_dump_folder = os.path.join(root, "processed", "val")

    if not os.path.exists(os.path.join(root, "processed", "test")):
        os.makedirs(os.path.join(root, "processed", "test"))
    test_dump_folder = os.path.join(root, "processed", "test")

    parameters = []
    for train_sub in train_e_sub:
        parameters.append([train_sub, train_dump_folder, 1])
    for train_sub in train_n_sub:
        parameters.append([train_sub, train_dump_folder, 0])
    for val_sub in val_e_sub:
        parameters.append([val_sub, val_dump_folder, 1])
    for val_sub in val_n_sub:
        parameters.append([val_sub, val_dump_folder, 0])
    for test_sub in test_e_sub:
        parameters.append([test_sub, test_dump_folder, 1])
    for test_sub in test_n_sub:
        parameters.append([test_sub, test_dump_folder, 0])

    with Pool(processes=24) as pool:
        result = pool.map(split_and_dump, parameters)


def main_pretraining_dataset():
    make_hdf5_for_MGB(savePath='/data/PRETRAIN_DATASET/', rewrite=True)
    add_hdf5_for_MGB(add_data_path='/data/SPIKES/segments_10min', num_files_to_select=500)
    add_hdf5_for_MGB(add_data_path='/data/MoE/events', num_files_to_select=2761)
    add_hdf5_for_MGB(add_data_path='/data/Sandor_100/EDF',eeg_type='edf', num_files_to_select=100)
    keys_without_eeg=check_HDF5_eeg_key()
    if len(keys_without_eeg)>1:
        print('[*] ATTENTION: some data in hdf5 do not have eeg key')
        print(keys_without_eeg)

def main_TUAB_finetuning_dataset():
    TUAB()


def main_TUEV_finetuning_dataset():
    TUEV()


def main_TUEP_finetuning_dataset():
    TUEP()




if __name__ == "__main__":
    main_pretraining_dataset()

    main_TUAB_finetuning_dataset()
    pass


