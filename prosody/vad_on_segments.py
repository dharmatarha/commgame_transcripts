"""

"""


import argparse
import torchaudio
import os
import sys
import pandas as pd
import re
import warnings
from glob import glob
import srt

sys.path.append('/home/gandalf/beast2/')

from audio_transcription_preproc import prepare_vad
from audio_transcription_preproc import wav_resample

# Known cases of missing audio in data set.
MISSING_AUDIO_DATA = [('pair100', 'BG4', 'Mordor'),
                      ('pair100', 'BG4', 'Gondor')]
# Audio resampling rate in Hz, resampling to this frequency is required by the VAD model.
RESAMPLING_RATE = 16000
# Params for vad model call
VAD_THRESHOLD = 0.90
VAD_MIN_SPEECH_DURATION_MS = 100
VAD_MIN_SILENCE_DURATION_MS = 100
VAD_WINDOW_SIZE_SAMPLES = 512
VAD_SPEECH_PAD_MS = 30
# Fix output name
DATAFRAME_SAVEFILE = 'speech_segment_VAD_results.pkl'
# Minimum audio length to consider for VAD
MIN_AUDIO_LENGTH_S = 0.05


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
    :param use_fixed:     Bool, flag for using audio segments derived from manually fixed transcriptions
                          ("..._segmentX.wav" files in subdirs ending with "..._fixed_segmentation").
    :return: audio_info:  Pandas df. Each row corresponds to one lab of one session of one pair.
    """

    df = pd.DataFrame(columns=['pair', 'session', 'lab', 'segments_dir', 'segments', 'fixed', 'speech_timestamps'])

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
        max_bg = int(max(res))

        sessions = ['BG' + str(idx) for idx in list(range(1, max_bg + 1))]
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

                # Check if current-pair - session - lab tuple is known as missing data
                current_data_tuple = ('pair'+str(current_pair), ses, lab)
                if current_data_tuple in MISSING_AUDIO_DATA:
                    pass
                # Otherwise, if the subdir exists, store the pair, session, lab and subdir info in the dataframe
                elif subdir:
                    df.loc[len(df)] = {'pair': current_pair, 'session': ses, 'lab': lab, 'segments_dir': subdir,
                                       'segments': [], 'fixed': fixed_flag, 'speech_timestamps': []}
                else:
                    error_str = 'Could not find subdir for pair' + str(current_pair) + ', session ' + ses +\
                                ', lab ' + lab + ' at ' + pair_dir + '!'
                    raise NotADirectoryError(error_str)

    # Go through all rows of dataframe and read in the segment wav file paths for each "segments_dir" value.

    # Regex pattern for segment audio files
    segment_regex = '.*_segment\d{1,2}'

    for row_idx in df.index:
        # Get the directory of the current row
        row_subdir = df.loc[row_idx, 'segments_dir']
        # Get all files matching the segment regex pattern in the subdir in a list, store it in 'segments' column of df.
        # Use .at for inserting a list into a cell.
        df.at[row_idx, 'segments'] = [f for f in os.listdir(row_subdir) if re.search(segment_regex, f)]

    return df


def srt_reader(srt_path):
    """
    Helper parsing subtitle start (s), end (s), and content  from srt file, using srt package.
    :param srt_path: Path to subtitle (srt) file.
    :return: start_times: List of subtitle start times.
    :return: end_times:   List of subtitle end times.
    :return: content:     List of subtitles as strings.
    """
    with open(srt_path, 'r') as f:
        srt_obj = srt.parse(f)
        subtitles = list(srt_obj)
    start_times = [s.start.total_seconds() for s in subtitles]
    end_times = [s.end.total_seconds() for s in subtitles]
    content = [s.content for s in subtitles]

    return start_times, end_times, content


def srt_timestamps_to_df(df, data_dir):
    """
    Function to find and load transcription files for each session, and extract speech-timing-related information.

    :param df:       Pandas dataframe holding speech-related info for each audio segment of each lab- and
                     session-specific recording. With columns "pair", "session", "lab", "fixed", "segments", etc.
    :param data_dir: Path to folder holding all transcription-related data, with subdirs "asr" and "fixed".
                     Srt files are expected to be in these subdirs.
    :return: df_srt: Pandas dataframe. Same as input df, but with added columns "srt_path and
                              "srt_segment_info".
    """

    df_srt = df.copy()

    df_srt['srt_path'] = pd.Series(dtype=str)
    df_srt['srt_segment_info'] = pd.Series(dtype=object)

    # Loop through the rows of the dataframe.
    for row in zip(df_srt.index, df_srt['pair'], df_srt['lab'], df_srt['session']):

        if row[0] % 50 == 0:
            print('Working on line ' + str(row[0]) + '...')

        # Define the naming pattern of the corresponding .srt file.
        srt_file_pattern = '_'.join(['pair' + str(row[1]), row[2], row[3],
                                     'repaired_mono_noisered_*.srt'])

        # Define the directory that should contain the .srt file we are looking for.
        if df_srt.loc[row[0], 'fixed']:
            srt_dir = os.path.join(data_dir, 'fixed', 'pair' + str(row[1]))
        else:
            srt_dir = os.path.join(data_dir, 'asr', 'pair' + str(row[1]))

        # Glob recursively for srt file, check return
        srt_path = glob(os.path.join(srt_dir, '**', srt_file_pattern), recursive=True)
        if not srt_path:
            raise FileNotFoundError(' '.join(['Missing file!', srt_file_pattern]))
        else:
            srt_path = srt_path[0]

        # Store path in df.
        df_srt.loc[row[0], 'srt_path'] = srt_path

        # Load srt file and extract speech segment timings
        start_times, end_times, content = srt_reader(srt_path)
        srt_segment_info = [{'start': z[0], 'end': z[1], 'content': z[2]}
                            for z in zip(start_times, end_times, content)]
        df_srt.at[row[0], 'srt_segment_info'] = srt_segment_info

    return df_srt


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
    parser.add_argument('--audio_dir',type=str,
                        default='/media/gandalf/data_hdd/audio_transcription/data/',
                        nargs='?',
                        help='Path to directory holding transcription-related files, including the audio segments, '
                             'with subdirs "/asr" and "/fixed".')
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
    print('Flag for manually fixed transcriptions: ' + str(args.use_fixed))
    print('\nLooking for relevant audio files...')

    # Find all files for the supplied pair number(s) and main data directory.
    # Returned var is a pandas dataframe
    df = find_audio(pair_numbers, os.path.join(args.audio_dir, 'asr'), use_fixed=args.use_fixed)
    print('Done with listing relevant files into dataframe:')
    print(df.head)

    # Prepare the VAD model
    vad_model, get_speech_timestamps = prepare_vad()

    # Loop through the rows of the dataframe, query the list of audio segments
    print('\nApplying VAD model to each speech segment, storing the results in dataframe...')
    for df_row_idx in df.index:
        row_audio_segments = df.loc[df_row_idx, 'segments']  # list
        row_segments_dir = df.loc[df_row_idx, 'segments_dir']  # path to dir with segment files

        # Loop through the audio segments, call resampling and the VAD model on each of them
        speech_timestamps = []
        for segment in row_audio_segments:
            segment_path = os.path.join(row_segments_dir, segment)

            # Check if audio is even long enough to call VAD model on it.
            info = torchaudio.info(segment_path)
            if info.num_frames > info.sample_rate * MIN_AUDIO_LENGTH_S:

                resampled_waveform = wav_resample(segment_path, RESAMPLING_RATE)
                segment_timestamps = get_speech_timestamps(resampled_waveform,
                                                          vad_model,
                                                          sampling_rate=RESAMPLING_RATE,
                                                          threshold=VAD_THRESHOLD,
                                                          min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
                                                          min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
                                                          window_size_samples=VAD_WINDOW_SIZE_SAMPLES,
                                                          speech_pad_ms=VAD_SPEECH_PAD_MS)
                speech_timestamps.append(segment_timestamps)

            else:
                warnings.warn(' '.join(['\nWAV FILE IS TOO SHORT FOR VAD:', segment_path]))
                speech_timestamps.append('')

        # Store list of timestamps in dataframe
        df.at[df_row_idx, 'speech_timestamps'] = speech_timestamps
    print('\nFinished with querying speech timestamps.')

    # Get speech segment information from srt files.
    df_srt = srt_timestamps_to_df(df, args.audio_dir)

    df_srt.to_pickle(os.path.join(args.audio_dir, DATAFRAME_SAVEFILE))
    print('Saved dataframe to', os.path.join(args.audio_dir, DATAFRAME_SAVEFILE))


if __name__ == '__main__':
    main()
