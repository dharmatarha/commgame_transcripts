"""

"""


import argparse
import torchaudio
import os
import sys
import pandas as pd
import re
import warnings
import copy
import srt
import numpy as np
from scipy import signal
from glob import glob

sys.path.append('/home/gandalf/beast2/')

from audio_transcription_preproc import prepare_vad
from audio_transcription_preproc import wav_resample

# Known cases of missing audio in data set.
MISSING_AUDIO_DATA = [('pair100', 'BG4', 'Mordor'),
                      ('pair100', 'BG4', 'Gondor')]
# Audio resampling rate in Hz, resampling to this frequency is required by the VAD model. This is also the frequency
# used earlier for e.g. the *_segmentation_samples.npz, the *_vad.json and other data files holding information
# in terms of audio samples.
RESAMPLING_RATE = 16000
# Params for vad model call
VAD_THRESHOLD = 0.90
VAD_MIN_SPEECH_DURATION_MS = 100
VAD_MIN_SILENCE_DURATION_MS = 100
VAD_WINDOW_SIZE_SAMPLES = 512
VAD_SPEECH_PAD_MS = 30
# Fix output names
DATAFRAME_SAVEFILE_INTERIM = 'speech_segment_VAD_results.pkl'
DATAFRAME_SAVEFILE_FINAL = 'speech_segment_results_all.pkl'
# Minimum audio length to consider for VAD
MIN_AUDIO_LENGTH_S = 0.05
# Target sampling rate for speech timeseries, in Hz
TIMESERIES_SAMPLING_RATE = 200


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


def digits_from_string_ends(str_list, max_d=3):
    """
    Helper to strip digits from the last "max_d" characters of strings. Strings are supplied in a list.
    Returns a list of stripped digits, as integers.
    :param str_list: List of strings
    :param max_d:    Int, the number of last characters to search for digits. Defaults to 3.
    :return: end_digits: List of integers.

    >>> a = ['abc1', 'dfg11', 'htk']
    >>> digits_from_string_ends(a, 2)
    [1, 11, None]
    """
    end_digits_str = [re.sub('[^0-9]', '', current_str[-max_d:]) for current_str in str_list]
    end_digits_int = [None if d == '' else int(d) for d in end_digits_str]
    return end_digits_int


def reorder_segments(df):
    """
    Helper function that sorts the lists in the "segments" column of the dataframe according to the numbers at
    the ends of file paths stored in each list. This ordering is also used to reorder the lists in the
    "speech_timestamps" column. The rest of the dataframe is left alone.

    :param df:          Pandas dataframe with columns "segments" and "speech_timestamps, each holding lists.
    :return: df_reord:  Same pandas dataframe as "df" but with the lists in "segments" and "speech_timestamps" reordered
                        according to the numbering of files in the "segments" lists.
    """
    df_reord = pd.DataFrame(columns=df.columns, data=copy.deepcopy(df.values))

    for row_idx in df.index:
        sequence = digits_from_string_ends(df_reord.loc[row_idx, 'segments'])
        seq_sorted, segments_sorted, timestamps_sorted = zip(*sorted(zip(sequence,
                                                                         df_reord.loc[row_idx, 'segments'],
                                                                         df_reord.loc[row_idx, 'speech_timestamps'])))
        df_reord.at[row_idx, 'segments'] = list(segments_sorted)
        df_reord.at[row_idx, 'speech_timestamps'] = list(timestamps_sorted)

    return df_reord


def srt_timestamps_to_df(df, data_dir):
    """
    Function to find and load transcription files for each session, and extract speech-timing-related information.

    :param df:       Pandas dataframe holding speech-related info for each audio segment of each lab- and
                     session-specific recording. With columns "pair", "session", "lab", "fixed", "segments", etc.
    :param data_dir: Path to folder holding all transcription-related data, with subdirs "asr" and "fixed".
                     Srt files are expected to be in these subdirs.
    :return: df_srt: Pandas dataframe. Same as input df, but with added columns "srt_path" and
                     "srt_segment_info".
    """

    df_srt = pd.DataFrame(columns=df.columns, data=copy.deepcopy(df.values))

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


def segment_ends_to_df(df):
    """
    Function to append the pandas dataframe with the segment timing information from the *_segmentation_samples.npz
    files.
    For each row (audio recording) in the dataframe where the transcription (srt) has not been fixed yet manually
    (that is, the "fixed" var is False), the corresponding *_segmentation_samples.npz is loaded and the start and end
    timestamps of the original segments is added to the dataframe, in a new column "segment_times".
    The value in "segment_times" is a list of dicts, where each dict has keys "start", "end", and "duration", and the
    corresponding values are in seconds.

    :param df:       Pandas dataframe with columns "segments_dir", "fixed", "pair", "lab", and "session".
    :return: df_seg: Pandas dataframe, same as "df" but with new column "segment_times".
    """
    df_seg = pd.DataFrame(columns=df.columns, data=copy.deepcopy(df.values))
    df_seg.loc[:, 'segment_times'] = None
    for row_idx in df_seg.index:
        # Segmentation endpoints are only useful for not-yet-fixed segmentations / subtitles
        if not df_seg.loc[row_idx, 'fixed']:
            # Segmentation endpoints are stored in an .npz file with fix path and naming
            seg_npz_path = os.path.join(df_seg.loc[row_idx, 'segments_dir'],
                                        '_'.join([os.path.split(df_seg.loc[row_idx, 'segments_dir'])[1],
                                                  'segmentation_samples.npz'])
                                        )
            # Load npz, extract audio samples, transform into timestamps
            seg_ends = np.load(seg_npz_path)
            starts = seg_ends['segment_starts'] / RESAMPLING_RATE
            ends = seg_ends['segment_ends'] / RESAMPLING_RATE
            durations = ends - starts
            # For each segment, store the timestamps and duration info in a dictionary, store these in a list.
            segment_times = [{'start': z[0], 'end': z[1], 'duration': z[2]} for z in zip(starts, ends, durations)]
            # Store the list in the dataframe.
            df_seg.at[row_idx, 'segment_times'] = segment_times

    return df_seg


def compare_segment_times_w_srt(df, tolerance_s=0.1):
    """
    Helper function that compares the content of columns "segment_times" and "srt_segment_info" in the pandas dataframe.
    Discrepancies are printed to the terminal.

    :param df:           Pandas dataframe.
    :param tolerance_s:  Numeric value, tolerance for comparing corresponding segment start and ending timestamps, in
                         seconds. Defaults to 0.1.
    :return: -
    """
    df_comp = pd.DataFrame(columns=df.columns, data=copy.deepcopy(df.values))

    for row_idx in df_comp.index:
        if not df_comp.loc[row_idx, 'fixed']:

            segment_times = df_comp.loc[row_idx, 'segment_times']
            srt_times = df_comp.loc[row_idx, 'srt_segment_info']

            for segment_idx, segment_current in enumerate(segment_times):
                if np.abs(segment_current['start'] - srt_times[segment_idx]['start']) > tolerance_s or \
                   np.abs(segment_current['end'] - srt_times[segment_idx]['end']) > tolerance_s:

                    srt_times.insert(segment_idx, {'start': segment_current['start'],
                                                   'end': segment_current['end'],
                                                   'content': ''})
                    print('Row ' + str(row_idx) + ': Inserted a dictionary with empty content into the ' +
                          'srt_segment_info list at index ' + str(segment_idx))

    return df_comp


def speech_timeseries(df, sampling_rate=None):
    """
    Helper function to generate speech time series for each set of segment speech timestamps in the input dataframe.
    Input pandas dataframe must have columns "speech_timestamps" and "srt_segment_info". Output is stored in new column
    "speech_timeseries", as a numpy array.

    :param df:            Pandas dataframe.
    :param sampling_rate: Integer, target sampling rate for downsampling the speech timeseries numpy array, in Hz.
                          Defaults to None, that is, the original sampling rate of RESAMPLING RATE used at the VAD
                          step is preserved.
    :return: df_ts: Pandas dataframe with the extra column "speech_timeseries" which holds a binary numpy array
                    marking speech as a time series.
    """
    df_ts = pd.DataFrame(columns=df.columns, data=copy.deepcopy(df.values))
    df_ts.loc[:, 'speech_timeseries'] = None

    # Loop through each row in dataframe / each audio recording
    for row_idx in df_ts.index:

        # Extract the two relevant variables, both of which is a list of dictionaries.
        srt_info = df_ts.loc[row_idx, 'srt_segment_info']
        vad_info = df_ts.loc[row_idx, 'speech_timestamps']
        # Sanity check - they should have the same length
        assert len(srt_info) == len(vad_info), 'Different length of srt-based and vad-based timing lists!'

        # Get the length of the timeseries from the last element in srt_info, round it to larger integer second.
        samples_no = np.ceil(srt_info[-1]['end']) * RESAMPLING_RATE
        # Initialize numpy array of zeros of length samples_no.
        speech_ts = np.zeros([int(samples_no)])

        # Loop through each dictionary in vad_info and srt_info, and fill the zero-array with ones where
        # speech was detected.
        for srt_idx in range(len(vad_info)):
            # Segments in srt define the wider range within which the VAD model tried to detect exact
            # boundaries of speech.
            srt_range_in_samples = (np.asarray([srt_info[srt_idx]['start'],
                                                srt_info[srt_idx]['end']]) * RESAMPLING_RATE).astype(int)
            # Given the srt-defined segment, go through the VAD-detected, potentially shorted speech segments within.
            if vad_info[srt_idx]:
                for vad_dict in vad_info[srt_idx]:
                    speech_range_in_samples = [vad_dict['start'] + srt_range_in_samples[0],
                                               vad_dict['end'] + srt_range_in_samples[0]]
                    # Set the timeseries to ones where there was speech.
                    speech_ts[speech_range_in_samples[0]: speech_range_in_samples[1]] = 1

        # The timeseries array can be downsampled if the sampling_rate argument was provided.
        if sampling_rate:
            assert sampling_rate < RESAMPLING_RATE, 'Requested sampling rate is higher than original!!!'
            assert (RESAMPLING_RATE/sampling_rate) % 1 == 0, 'The ratio of requested and original sampling rate is not integer!!!'
            speech_ts_downsamp = signal.decimate(speech_ts, int(RESAMPLING_RATE/sampling_rate), ftype='fir')
            # Turn it back into binary integer array
            speech_ts_downsamp[speech_ts_downsamp < 0.5] = 0
            speech_ts_downsamp[speech_ts_downsamp > 0.5] = 1
            speech_ts = speech_ts_downsamp.astype(int)

        # Store timeseries in dataframe
        df_ts.at[row_idx, 'speech_timeseries'] = speech_ts

    return df_ts


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

        # User feedback.
        if df_row_idx % 10 == 0:
            print('\nWorking on row ' + str(df_row_idx))

        # Select relevant variables from dataframe.
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

    # Reorder the "segments" and "speech_timestamps" columns according to the sequence numbers in the segment filenames.
    df = reorder_segments(df)

    # Get speech segment information from srt files.
    df_srt = srt_timestamps_to_df(df, args.audio_dir)

    # Add segment timing info for rows where the transcriptions have not yet been manually "fixed".
    df_seg = segment_ends_to_df(df_srt)

    # Repair mismatches between srt_segment_info lists and segment_times lists. That is, when some segments
    # generated no speech content in the srt.
    df_comp = compare_segment_times_w_srt(df_seg, tolerance_s=0.1)

    # Interim save, just to be sure
    df_comp.to_pickle(os.path.join(args.audio_dir, DATAFRAME_SAVEFILE_INTERIM))
    print('Saved dataframe to', os.path.join(args.audio_dir, DATAFRAME_SAVEFILE_INTERIM))

    # Get timeseries data for each row of the dataframe
    df_ts = speech_timeseries(df_comp, sampling_rate=TIMESERIES_SAMPLING_RATE)

    df_ts.to_pickle(os.path.join(args.audio_dir, DATAFRAME_SAVEFILE_FINAL))
    print('Saved dataframe to', os.path.join(args.audio_dir, DATAFRAME_SAVEFILE_FINAL))


if __name__ == '__main__':
    main()
