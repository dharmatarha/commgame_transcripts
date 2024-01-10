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


def find_wavs(pair_numbers, audio_dir):
    res = {}
    for no in pair_numbers:
        pattern = os.path.join(audio_dir, '**/pair' + str(no) + '_*_repaired_mono.wav')
        res[no] = glob(pattern, recursive=True)
        if not res[no]:
            raise UserWarning(f'There were no wav files for pair {no}!')
    return res


def dataframe_from_wav_dict(wav_files_dict):
    df_columns = ['wav', 'pair']
    df = pd.DataFrame(columns=df_columns)
    for key in wav_files_dict.keys():
        pair_no = [key] * len(wav_files_dict[key])
        pair_df = pd.DataFrame(list(zip(pair_no, wav_files_dict[key])),
                               columns=df_columns)
        df = pd.concat([df, pair_df], axis=0, ignore_index=True)
    return df


def get_audio_timing(df):
    """
    Input is a pandas dataframe with at least the column "wav" containing paths to wav files.
    A new column named 'wav_len' is added with the length of each wav file in seconds.
    """
    wav_col_idx = df.columns.get_loc('wav')
    df.insert(wav_col_idx + 1, 'wav_len', ['nan'] * len(df))
    for idx in df.index:
        wav_file = df['wav'][idx]
        y, sr = librosa.load(wav_file, sr=None)
        df['wav_len'][idx] = len(y)/sr
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
        session_name = basename.split('_')[1]
        if session_name.startswith('BG'):
            session_type = 'BG'
        elif session_name.startswith('freeConv'):
            session_type = 'freeConv'
        else:
            session_type = 'nan'
        df['session'][idx] = session_type
    return df


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

    csv_path = os.path.join(args.audio_dir, 'wav_times.csv')
    df.to_csv(csv_path)
    print('\nSaved dataframe into csv at', csv_path)

    return


if __name__ == '__main__':
    main()
