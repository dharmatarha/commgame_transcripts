"""

"""


import argparse
import torchaudio
import torch
import os
import numpy as np
import sys
import pandas as pd
import re

sys.path.append('/home/adamb/commgame_transcripts/vad_segment')

from audio_transcription_preproc import prepare_vad
from audio_transcription_preproc import wav_resample


def find_audio(pair_number, data_dir, use_fixed=False):
    """
    Find all segment wav files for a given pair number in the directory specified by "data_dir".
    Dir "data_dir" must contain these files in pair-specific subdirs, which in turn contain
    lab- and session-specific subdirs, e.g. "pair99_Mordor_BG1_repaired_mono_noisered".
    Different subdir endings are valid:
    "..._repaired_mono_noisered",
    "..._repaired_mono_noisered_filtered",
    "..._fixed_segmentation" (if the option use_fixed is set to True).
    The audio files conform to one of the following formats, depending on their origin:
        "pair[PAIR_NO]_[Mordor|Gondor]_[freeConv|BGx]_repaired_mono_noisered_segment[y].wav"
        "pair[PAIR_NO]_[Mordor|Gondor]_[freeConv|BGx]_repaired_mono_nosiered_filtered_segment[y].wav"

    :param pair_number:   List of integers, pair numbers.
    :param data_dir:      Str, path to directory containing audio files in pair-specific subdirs.
    :param filtered:      Bool, flag for using filtered audio for "noise_reduced" files
                          (that is, wav files with "_filtered.wav" ending).
    :return: audio_info:  Pandas df. Each row corresponds to one lab of one session of one pair.
    """

    df = pd.DataFrame(columns=['pair', 'session', 'lab', 'segments_dir', 'segments', 'fixed', 'speech'])

    for current_pair in pair_number:

        # Regex pattern for BG game numbers for current pair
        bg_games_pattern = '^pair' + str(current_pair) + '_(Mor|Gon)dor_BG\d'

        # Define pair-level subdir
        pair_dir = os.path.join(data_dir, 'pair' + str(current_pair))
        if not os.path.exists(pair_dir):
            raise NotADirectoryError

        # Check for session- and lab-specific subdirs
        dir_content = os.listdir(pair_dir)
        # Determine the highest BG game number available. First, get the number after the "BG" part of the strings,
        # then find the maximum.
        res = [s[re.search(bg_games_pattern, s).end() - 1] for s in dir_content if re.search(bg_games_pattern, s)]
        max_bg = max(res)

        sessions = ['BG' + idx for idx in list(range(1, max_bg + 1))]
        sessions.append('freeConv')

        # Go through all possible sessions, and look for corresponding subdirs, separately for each lab.
        # If the use_fixed option is set, first try to look for subdirs with "fixed" segments, then try "*_filtered",
        # and finally the standard "*_noisered"
        for ses in sessions:

            for lab in ['Mordor', 'Gondor']:
                fixed_flag = False
                subdir = []

                subdir_fixed = os.path.join(pair_dir, '_'.join(['pair' + str(current_pair), lab,
                                                                ses, 'fixed_segmentation']))
                subdir_filtered = os.path.join(pair_dir, '_'.join(['pair' + str(current_pair), lab,
                                                                   ses, 'repaired_mono_noisered_filtered']))
                subdir_base = os.path.join(pair_dir, '_'.join(['pair' + str(current_pair), lab,
                                                               ses, 'repaired_mono_noisered']))
                if use_fixed and os.path.exists(subdir_fixed):
                    subdir = subdir_fixed
                    fixed_flag = True
                elif os.path.exists(subdir_filtered):
                    subdir = subdir_filtered
                elif os.path.exists(subdir_base):
                    subdir = subdir_base

                if subdir:
                    df.loc[len(df)] = {'pair': current_pair, 'session': ses, 'lab': lab, 'segments_dir': subdir,
                                       'segments': [], 'fixed': fixed_flag, 'speech': []}
                else:
                    error_str = 'Could not find subdir for pair' + str(current_pair) + ', session ' + ses +\
                                ', lab ' + lab + ' at ' + pair_dir + '!'
                    raise NotADirectoryError(error_str)

        # Go through all rows of dataframe and read in the segment wav file paths for each "segments_dir" value.


    return df



def main():
    # Input argument handling
    # One mandatory and two optional arguments, defining pair number(s), the path to the audio segments, and setting
    # the option for working with the re-generated segments after manual fixes to the transcripts (srt-s).
    # Arg args.pair_numbers is a list with one or more elements, args.audio_dir is string (path to dir), and
    # args.use_fixed is boolean.
    parser = argparse.ArgumentParser()
    parser.add_argument('pair_numbers', type=int, nargs='+',
                        help='Pair numbers, determines which audio (wav) files are selected for analysis.'
                             'If exactly two numbers are provided, a range of pair numbers is defined by treating '
                             'the two numbers as the first and last pair numbers.')
    parser.add_argument('--audio_dir',
                        type=str,
                        default='/media/gandalf/data_hdd/audio_transcription/data/asr/',
                        nargs='?',
                        help='Path to directory holding the audio segments (short wav files), '
                             'with subdirectories for each pair.')
    parser.add_argument('--use_fixed', action='store_true',
                        help='Flag for using segments derived from manually fixed srt-s '
                             '(wav files in session-specific dirs with "fixed_segmentation" ending).')
    args = parser.parse_args()

    if not os.path.exists(args.audio_dir):
        raise NotADirectoryError('Input arg --audio_dir is not a valid path!')

    # If there are two pair numbers, expand it into a range
    if len(args.pair_numbers) == 2:
        pair_numbers = list(range(args.pair_numbers[0], args.pair_numbers[1]+1))
    else:
        pair_numbers = args.pair_numbers

    print('\nCalled vad_on_segments with args:')
    print('Pair numbers: ' + str(pair_numbers))
    print('Audio directory: ' + args.audio_dir)
    print('Flag for filtered audio input: ' + str(args.use_fixed))
    print('\nLooking for relevant audio files...')

    # Find all files for the supplied pair number(s) and main data directory.
    # Returned var is a pandas dataframe
    df = find_audio(pair_numbers, args.audio_dir, use_fixed=args.use_fixed)


if __name__ == '__main__':
    main()
