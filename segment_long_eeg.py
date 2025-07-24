import os
import mne
import numpy as np
from pathlib import Path
from tqdm import tqdm
import hdf5storage
import mat73
import scipy
import argparse
from collections import defaultdict
import pandas as pd
import re




def split_edf_files(input_folder, output_folder, segment_duration=600):
    """
    Split all EDF files in a folder into segments of specified duration using MNE

    Parameters:
    input_folder (str): Path to input folder containing EDF files
    output_folder (str): Path to output folder for segments
    segment_duration (int): Duration of each segment in seconds, default 600s (10 minutes)
    """

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Find all EDF files
    edf_files = []
    for ext in ['*.edf',
                # '*.EDF'
                ]:
        edf_files.extend(Path(input_folder).glob(ext))

    if not edf_files:
        print(f"No EDF files found in {input_folder}")
        return

    print(f"Found {len(edf_files)} EDF files")

    # Set MNE log level to WARNING to reduce output
    mne.set_log_level('WARNING')

    for edf_path in tqdm(edf_files, desc='Segmenting EDFs'):
        try:
            split_single_edf(str(edf_path), output_folder, segment_duration)
        except Exception as e:
            print(f"Error processing file {edf_path.name}: {str(e)}")
            continue


def split_single_edf(input_file, output_folder, segment_duration=600):
    """
    Split a single EDF file using MNE

    Parameters:
    input_file (str): Path to input EDF file
    output_folder (str): Path to output folder
    segment_duration (int): Duration of each segment in seconds
    """

    # Read EDF file using MNE
    try:
        raw = mne.io.read_raw_edf(input_file, preload=True, verbose=False)
    except Exception as e:
        print(f"  Cannot read file: {str(e)}")
        return

    # Get basic information
    total_duration = raw.times[-1]  # Total duration in seconds

    # Calculate number of segments needed
    n_segments = int(np.ceil(total_duration / segment_duration))

    # Get base filename without extension
    base_name = Path(input_file).stem

    # Split and save each segment
    for segment_idx in range(n_segments):
        # Calculate time range for current segment
        start_time = segment_idx * segment_duration
        end_time = min(start_time + segment_duration, total_duration)

        # Extract time segment using MNE's crop method
        raw_segment = raw.copy().crop(tmin=start_time, tmax=end_time)

        # Generate output filename
        output_filename = f"{base_name}_{segment_idx}.edf"
        output_path = os.path.join(output_folder, output_filename)

        # Save segment
        try:
            mne.export.export_raw(output_path, raw_segment, fmt='edf', overwrite=True, verbose=False)

        except Exception as e:
            print(f"    Error saving segment {segment_idx + 1}: {str(e)}")
            continue


def load_mat_file(file_path):
    """
    Load MAT file using multiple loaders for different MAT versions

    Parameters:
    file_path (str): Path to MAT file

    Returns:
    dict: Loaded MAT data
    """

    try:
        raw = mat73.loadmat(file_path)
        signal = raw['data']
        self_fs = get_frequency_from_mat(raw_mat=raw)
        channel_names = get_channel_names_from_mat(raw_mat=raw)
        return raw, signal, self_fs, channel_names, 'mat73'

    except TypeError:
        try:
            raw = scipy.io.loadmat(file_path)
            signal = raw['data']
            self_fs = get_frequency_from_mat(raw_mat=raw)
            channel_names = get_channel_names_from_mat(raw_mat=raw)
            return raw, signal, self_fs, channel_names, 'scipy'

        except Exception as e:
            try:
                raw = hdf5storage.loadmat(file_path)
                signal = raw['data']
                self_fs = get_frequency_from_mat(raw_mat=raw)
                channel_names = get_channel_names_from_mat(raw_mat=raw)
                return raw, signal, self_fs, channel_names, 'hdf5storage'

            except Exception as e:
                raise ValueError(f'Failed to load {file_path}. Mat type error : {e}')


def get_channel_names_from_mat(raw_mat):
    """
    Extract channel names from the channels array and remove leading/trailing spaces.
    :param channels: Channels array
    :return: List of channel names
    """
    try:
        channels = raw_mat['channels']
    except KeyError:
        try:
            channels = raw_mat['channel_locations']
        except KeyError:
            raise ValueError(f'No channel names found in mat')

    channel_names = []
    # Iterate through the channels array
    for channel in channels:
        # Handle different cases
        if isinstance(channel, np.ndarray):
            # If channel is a nested array
            if channel.size == 1:
                channel_name = channel.item()
            else:
                channel_name = channel[0]

            # Further unwrap if necessary
            if isinstance(channel_name, np.ndarray):
                if channel_name.size == 1:
                    channel_name = channel_name.item()
                else:
                    channel_name = channel_name[0]

                # Further unwrap if necessary
                if isinstance(channel_name, np.ndarray):
                    if channel_name.size == 1:
                        channel_name = channel_name.item()
                    else:
                        channel_name = channel_name[0]
        elif isinstance(channel, list):
            # If channel is a list
            channel_name = channel[0]
        else:
            # If channel is a single element
            channel_name = channel

        # Ensure the channel name is a string
        if isinstance(channel_name, np.ndarray):
            channel_name = channel_name.item()
        channel_names.append(channel_name.strip())

    return channel_names


def get_frequency_from_mat(raw_mat):
    try:
        fs_value = raw_mat['Fs']
    except KeyError:
        try:
            fs_value = raw_mat['fs']
        except KeyError:
            try:
                fs_value = raw_mat['sampling_rate']
            except KeyError:
                return 0

    if isinstance(fs_value, np.ndarray):
        if fs_value.shape == (1, 1, 1):
            fs_value = fs_value[0, 0, 0]
        elif fs_value.shape == (1, 1):
            fs_value = fs_value[0, 0]
        elif fs_value.shape == (1,):
            fs_value = fs_value[0]
        elif fs_value.shape == ():
            fs_value = fs_value.item()
        else:
            print('Unexpected array shape for fs value in mat')
            return 0

    if isinstance(fs_value, np.ndarray):
        fs_value = fs_value.item()

    return int(fs_value)


def split_mat_files(input_folder, output_folder, segment_duration=600, sampling_rate=None):
    """
    Split all MAT files in a folder into segments of specified duration

    Parameters:
    input_folder (str): Path to input folder containing MAT files
    output_folder (str): Path to output folder for segments
    segment_duration (int): Duration of each segment in seconds, default 600s (10 minutes)
    data_key (str): Key name for data in MAT file, default 'data'
    fs_key (str): Key name for sampling rate in MAT file. If None, auto-detect
    sampling_rate (float): Manual sampling rate if not found in file
    """

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Find all MAT files
    mat_files = []
    for ext in ['*.mat']:
        mat_files.extend(Path(input_folder).glob(ext))

    if not mat_files:
        print(f"No MAT files found in {input_folder}")
        return

    for mat_path in tqdm(mat_files, desc='Segmenting MAT files'):
        try:
            split_single_mat(str(mat_path), output_folder, segment_duration,
                             sampling_rate)
        except Exception as e:
            print(f"Error processing file {mat_path.name}: {str(e)}")
            continue


def split_single_mat(input_file, output_folder, segment_duration=600, sampling_rate=None):
    """
    Split a single MAT file using multiple loaders

    Parameters:
    input_file (str): Path to input MAT file
    output_folder (str): Path to output folder
    segment_duration (int): Duration of each segment in seconds
    data_key (str): Key name for data in MAT file
    fs_key (str): Key name for sampling rate in MAT file
    sampling_rate (float): Manual sampling rate if not found in file
    """

    # Load MAT file using multiple loaders
    try:
        raw, signal, signal_fs, channel_names, loader_used = load_mat_file(input_file)
    except Exception as e:
        print(f"  Cannot read MAT file: {str(e)}")
        return

    # Remove MATLAB metadata keys if they exist (only for scipy loader)

    if signal_fs == 0:
        if sampling_rate:
            signal_fs = sampling_rate
        else:
            signal_fs = 200
            print(f'{input_file} has no sampling rate in data file or input parameter, using 200 Hz')

    n_channels, n_samples = signal.shape

    # Calculate number of segments
    samples_per_segment = int(segment_duration * signal_fs)
    n_segments = int(np.ceil(n_samples / samples_per_segment))

    # Get base filename without extension
    base_name = Path(input_file).stem

    # Split and save each segment
    for segment_idx in range(n_segments):
        start_sample = segment_idx * samples_per_segment
        end_sample = min(start_sample + samples_per_segment, n_samples)

        # Extract segment data
        segment_data = signal[start_sample:end_sample, :]

        segment_dict = {}
        segment_dict['data'] = segment_data
        segment_dict['Fs'] = signal_fs
        segment_dict['channels'] = channel_names

        # Generate output filename
        output_filename = f"{base_name}_{segment_idx}.mat"
        output_path = os.path.join(output_folder, output_filename)

        # Save segment using scipy (most compatible)
        try:
            scipy.io.savemat(output_path, segment_dict)
        except Exception as e:
            print(f"    Error saving segment {segment_idx + 1}: {str(e)}")
            continue


def combine_result_files(input_folder, is_second_step, output_folder=None, sort_by_index=True):
    """
    Read CSV files from a folder and combine them based on filename index

    Filename format: prefix_index.csv
    Example: data_file_0.csv, data_file_1.csv, data_file_2.csv -> combined as data_file.csv

    Parameters:
    input_folder (str): Input folder path
    output_folder (str): Output folder path, if None output to input folder
    sort_by_index (bool): Whether to sort by index before combining, default True

    Returns:
    dict: Statistics of combination results
    """

    input_path = Path(input_folder)
    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")

    # Set output folder
    if output_folder is None:
        output_path = input_path
    else:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = list(input_path.glob('*.csv'))
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return {}

    # Group files by prefix
    file_groups = defaultdict(list)

    for csv_file in csv_files:
        filename = csv_file.stem  # Get filename without extension

        # Split filename by underscore
        parts = filename.split('_')

        # Check if the last part is a number (index)
        try:
            index = int(parts[-1])
            prefix = '_'.join(parts[:-1])  # Use preceding parts as prefix

            file_groups[prefix].append({
                'file_path': csv_file,
                'index': index,
                'filename': csv_file.name
            })

        except ValueError:
            print(f"Warning: Last part of filename is not a number, skipping: {csv_file.name}")
            continue

    if not file_groups:
        print("No files found matching the naming convention")
        return {}

    for prefix, file_list in tqdm(file_groups.items()):

        # Sort by index
        if sort_by_index:
            file_list.sort(key=lambda x: x['index'])

        # Read and combine CSV files
        dataframes = []

        for file_info in file_list:
            try:
                df = pd.read_csv(file_info['file_path'])
                dataframes.append(df)

            except Exception as e:
                print(f"  Error: Failed to read file {file_info['filename']}: {str(e)}")
                continue

        if not dataframes:
            print(f"  Warning: No files successfully read for group {prefix}")
            continue

        # Concatenate dataframes
        try:
            combined_df = pd.concat(dataframes, ignore_index=True)

            # Save combined file
            output_filename = f"{prefix}.csv"
            output_file_path = output_path / output_filename

            combined_df.to_csv(output_file_path, index=False)


        except Exception as e:
            print(f"  Error: Failed to combine group {prefix}: {str(e)}")
            continue

    print(f"\nCombination completed! Processed {len(file_groups.items())} file groups")
    return {}



# def get_args_for_segment():
#     """
#     Parse command line arguments for EDF/MAT file splitting
#
#     Returns:
#     argparse.Namespace: Parsed arguments
#     """
#     parser = argparse.ArgumentParser(
#         description='Split EDF/MAT files into segments of specified duration',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
#         Examples:
#           # Split EDF files
#           python script.py --eeg_dir input_folder --eval_sub_dir output_folder
#
#           # Split MAT files with custom sampling rate
#           python script.py --eeg_dir input_folder --eval_sub_dir output_folder --sampling_rate 600
#
#           # Use custom segment duration (5 minutes)
#           python script.py --eeg_dir data_folder --eval_sub_dir segments --segment_duration 300
#                 """
#     )
#     parser.add_argument(
#         '--data_format',
#         type=str,
#         required=True,
#         help='Input EEG format, should be EDF/MAT files'
#     )
#
#     parser.add_argument(
#         '--eeg_dir',
#         type=str,
#         required=True,
#         help='Path to input file or folder containing EDF/MAT files'
#     )
#
#     parser.add_argument(
#         '--eval_sub_dir',
#         type=str,
#         required=True,
#         help='Path to output folder for segments'
#     )
#
#     parser.add_argument(
#         '--sampling_rate',
#         type=float,
#         default=None,
#         help='Manual sampling rate (Hz) for MAT files if not found in file (default: None)'
#     )
#
#     parser.add_argument(
#         '--segment_duration',
#         type=int,
#         default=600,
#         help='Duration of each segment in seconds (default: 600s = 10 minutes)'
#     )
#
#     return parser.parse_args()
#
# def get_args_for_combine():
#
#     parser = argparse.ArgumentParser(
#         description='Combine EEG\'s segment results',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#
#     parser.add_argument(
#         '--eval_results_dir',
#         type=str,
#         required=True,
#         help='The morgoth results folder'
#     )
#
#     parser.add_argument(
#         '--prediction_slipping_step',
#         type=int,
#         default=None,
#         help='Slipping step (point unit) in continuous prediction'
#     )
#
#     parser.add_argument(
#         '--prediction_slipping_step_second',
#         type=int,
#         default=None,
#         help='Slipping step (second unit) in continuous prediction'
#     )
#
#     return parser.parse_args()


def get_args():
    """
    Main argument parser with subcommands

    Returns:
    argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='EEG file processing toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Available commands:
          segment    Split EDF/MAT files into segments
          combine    Combine EEG segment results
        
        Examples:
          # Segment files
          python script.py segment --data_format EDF --eeg_dir input_folder --eval_sub_dir output_folder
        
          # Combine results
          python script.py combine --eval_results_dir results_folder --prediction_slipping_step 100
                """
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Segment subcommand
    segment_parser = subparsers.add_parser(
        'segment',
        help='Split EDF/MAT files into segments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    segment_parser.add_argument(
        '--data_format',
        type=str,
        required=True,
        choices=['EDF', 'MAT', 'edf', 'mat'],
        help='Input EEG format, should be EDF/MAT files'
    )
    segment_parser.add_argument(
        '--eeg_dir',
        type=str,
        required=True,
        help='Path to input file or folder containing EDF/MAT files'
    )
    segment_parser.add_argument(
        '--eval_sub_dir',
        type=str,
        required=True,
        help='Path to output folder for segments'
    )
    segment_parser.add_argument(
        '--sampling_rate',
        type=float,
        default=None,
        help='Manual sampling rate (Hz) for MAT files if not found in file (default: None)'
    )
    segment_parser.add_argument(
        '--segment_duration',
        type=int,
        default=600,
        help='Duration of each segment in seconds (default: 600s = 10 minutes)'
    )

    # Combine subcommand
    combine_parser = subparsers.add_parser(
        'combine',
        help='Combine EEG segment results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    combine_parser.add_argument(
        '--eval_results_dir',
        type=str,
        required=True,
        help='The morgoth results folder'
    )

    combine_parser.add_argument(
        '--combined_results_dir',
        type=str,
        default=None,
        help='The combine results output folder'
    )

    combine_parser.add_argument(
        '--prediction_slipping_step',
        type=int,
        default=None,
        help='Slipping step (point unit) in continuous prediction'
    )
    combine_parser.add_argument(
        '--prediction_slipping_step_second',
        type=int,
        default=None,
        help='Slipping step (second unit) in continuous prediction'
    )

    return parser.parse_args()

def main():
    """
    Main function to handle command line arguments and execute file splitting
    """
    # Parse command line arguments
    args = get_args()

    if args.command == 'segment':

        data_format=args.data_format

        input_dir = args.eeg_dir

        output_dir = args.eval_sub_dir

        segment_duration= args.segment_duration

        sampling_rate=args.segment_duration

        if data_format=='mat':
            split_mat_files(input_folder=input_dir, output_folder=output_dir, segment_duration=segment_duration, sampling_rate=sampling_rate)

        elif data_format=='edf':
            split_edf_files(input_folder=input_dir, output_folder=output_dir, segment_duration=segment_duration)

        else:
            print('Only support mat or edf format')

    elif args.command == 'combine':
       ##########to do
        results_dir=args.eval_results_dir

        combined_results_dir=args.combined_results_dir

        prediction_slipping_step=args.prediction_slipping_step

        prediction_slipping_step_second=args.prediction_slipping_step_second


        if prediction_slipping_step_second is not None:
            combine_result_files(input_folder=results_dir,
                             is_second_step=True,
                             output_folder=combined_results_dir,
                             sort_by_index=True)

        elif prediction_slipping_step is not None:
            combine_result_files(input_folder=results_dir,
                                 is_second_step=False,
                                 output_folder=combined_results_dir,
                                 sort_by_index=True)

        else:
            print('Should provide prediction_slipping_step or prediction_slipping_step_second')

    else:
        print("Available commands: segment, combine")


if __name__ == "__main__":
    main()