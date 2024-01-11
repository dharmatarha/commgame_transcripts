""""
Utility to query details about the length of each session for a given pair in the CommGame dataset

USAGE:
python commgame_times_from_wavs.py PAIRNO [PAIRNO2 PAIRNO3 ...] --audio_dir PATH_TO_WAVS

Timing data is read from the CommGame audio recordings. Input arg "--audio_dir" must point to the folder containing
the necessary wav files which follow the naming convention "pairPAIRNO_LAB_SESSION_repaired_mono.wav". The folder is
globbed recursively for the necessary files.

"""

from glob import glob
import argparse
import librosa
import os
import pandas as pd
import warnings
import csv


def find_wavs(pair_numbers, audio_dir):
    res = {}
    for no in pair_numbers:
        pattern = os.path.join(audio_dir, '**/pair' + str(no) + '_*_repaired_mono.wav')
        res[no] = glob(pattern, recursive=True)
        if not res[no]:
            warnings.warn(f'There were no wav files for pair {no}!')
    return res


def dataframe_from_wav_dict(wav_files_dict):
    df_columns = ['wav', 'pair']
    df = pd.DataFrame(columns=df_columns)
    for key in wav_files_dict.keys():
        pair_no = [key] * len(wav_files_dict[key])
        pair_df = pd.DataFrame(list(zip(wav_files_dict[key], pair_no)),
                               columns=df_columns)
        df = pd.concat([df, pair_df], axis=0, ignore_index=True)
    return df


def get_audio_timing(df):
    """
    Input is a pandas dataframe with at least the column "wav" containing paths to wav files.
    A new column named 'wav_len' is added with the length of each wav file in seconds.
    """
    df.insert(len(df.columns), 'wav_len', ['nan'] * len(df))
    for idx in df.index:
        wav_file = df['wav'][idx]
        y, sr = librosa.load(wav_file, sr=None)
        df['wav_len'][idx] = round(len(y)/sr, 3)
    return df


def add_session_info(df):
    """
    Add session type column to input dataframe and populate it.
    Strictly depends on the file naming convention used in the CommGame project.
    """
    df.insert(len(df.columns), 'session', ['nan'] * len(df))
    for idx in df.index:
        wav_file = df['wav'][idx]
        basename = os.path.split(wav_file)[1]
        session_name = basename.split('_')[2]
        if session_name.startswith('BG'):
            session_type = 'BG'
        elif session_name.startswith('freeConv'):
            session_type = 'freeConv'
        else:
            session_type = 'nan'
        df['session'][idx] = session_type
    return df


def add_lab_info(df):
    """
    Add lab (Mordor or Gondor) column to input dataframe and populate it.
    Strictly depends on the file naming convention used in the CommGame project.
    """
    df.insert(len(df.columns), 'lab', ['nan'] * len(df))
    for idx in df.index:
        wav_file = df['wav'][idx]
        basename = os.path.split(wav_file)[1]
        lab_name = basename.split('_')[1]
        if lab_name not in ['Mordor', 'Gondor']:
            lab_name = 'nan'
            warnings.warn(f'Wrong lab name in file {wav_file}!')
        df['lab'][idx] = lab_name
    return df


def extract_pair_data(df):
    """
    Collect wav length data for each pair.
    """
    # Transform the "pair" column of the dataframe to a list
    pairs = df['pair'].tolist()
    # Get unique values
    pairs = set(pairs)
    pairs = list(pairs)

    # For each pair number, collect all timing data from dataframe, separately for BG and freeConv session types.
    timing_data = []
    for p in pairs:
        # Get dataframe row indices for current pair number, BG session, only for lab Mordor.
        bg_df = df.loc[(df['pair'] == p) & (df['lab'] == 'Mordor') & (df['session'] == 'BG')]
        # Sum corresponding wav length values
        pair_bg_time = bg_df['wav_len'].sum()
        # Get dataframe row indices for current pair number, freeConv session, only for lab Mordor.
        freeconv_df = df.loc[(df['pair'] == p) & (df['lab'] == 'Mordor') & (df['session'] == 'freeConv')]
        # Sum corresponding wav length values
        pair_freeconv_time = freeconv_df['wav_len'].sum()
        # Collect pair-specific timing data into a tuple, append it to the output list.
        timing_data.append((p, pair_bg_time, pair_freeconv_time, pair_bg_time + pair_freeconv_time))

    return timing_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('pair_numbers', type=int, nargs='+',
                        help='Pair numbers, determines which audio (wav) files are selected for analysis.'
                             'If exactly two numbers are provided, a range of pair numbers is defined by treating '
                             'the two numbers as the first and last pair numbers.')
    parser.add_argument('--audio_dir', type=str, default='/media/adamb/data_disk/CommGame/', nargs='?',
                        help='Path to folder containing the wav files. Defaults to /media/adamb/data_disk/CommGame/.')
    args = parser.parse_args()

    # If there are two pair numbers, expand it into a range
    if len(args.pair_numbers) == 2:
        pair_numbers = list(range(args.pair_numbers[0], args.pair_numbers[1]+1))
    else:
        pair_numbers = args.pair_numbers

    # Find all wav files for each pair matching the hardcoded pattern in find_wavs().
    wav_files_dict = find_wavs(pair_numbers, args.audio_dir)
    print('\nListed all wav files for pairs', pair_numbers)

    # Construct a pandas dataframe from each key-value pair in wav_files_dict and concatenate them.
    df = dataframe_from_wav_dict(wav_files_dict)
    print('\nConstructed a dataframe from pair numbers and wav paths.')

    # Add wav length in seconds to each row in dataframe.
    print('\nQuerying the length of each wav file, might take a bit...')
    df = get_audio_timing(df)
    print('\nQueried the length of each wav file.')

    # Add session type to each row in dataframe.
    df = add_session_info(df)
    print('\nAdded session type for each wav file.')

    # Add lab name to each row in dataframe.
    df = add_lab_info(df)
    print('\nAdded lab name for each wav file.')

    # Get summary timing data for session types, per pair.
    timing_data = extract_pair_data(df)  # timing_data is list of tuples

    # Write out dataframe as csv.
    csv_path = os.path.join(args.audio_dir, 'wav_times.csv')
    df.to_csv(csv_path)
    print('\nSaved dataframe into csv at', csv_path)

    # Write out timing_data as csv.
    csv_path = os.path.join(args.audio_dir, 'summary_wav_times.csv')
    with open(csv_path, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['pair', 'bg_time', 'freeconv_time', 'sum'])
        csv_out.writerows(timing_data)
    print('\nSaved summary timing data into csv at', csv_path)

    return


if __name__ == '__main__':
    main()
