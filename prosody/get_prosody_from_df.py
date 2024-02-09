"""

"""

import pandas as pd
import parselmouth as pm
import numpy as np
import os
import copy
import argparse
import pickle
import re


RESAMPLING_RATE = 16000
PM_TIME_STEP = 0.025
PM_MINIMUM_PITCH_HZ = 75
PM_MAXIMUM_PITCH_HZ = 750
PM_VOICING_THRESHOLD = 0.50
PM_PITCH_ACCURATE = True
MIN_SEGMENT_LENGTH_S = round(6.4/PM_MINIMUM_PITCH_HZ + 0.05, 1)
TIMESERIES_SAMPLING_RATE_HZ = 200
INPUT_PICKLE_FILENAME = 'speech_segment_results_all.pkl'  # Final output of vad_on_segments.py
PITCH_INTENSITY_PICKLE_FILENAME = 'prosody_timeseries_df.pkl'
INTENSITY_PKL_FILENAME = 'intensity_timeseries.pkl'
PITCH_PKL_FILENAME = 'pitch_timeseries.pkl'
SYLLABLE_PKL_FILENAME = 'syllable_timeseries.pkl'
PITCH_INTENSITY_SPEECHVARS_PICKLE_FILENAME = 'prosody_w_speech_df.pkl'


def add_speech_vars_to_df(df):
    """
    Helper function to derive a number of summary variables related to the amount of speech and to speech rate, and to
    add them to the input pandas dataframe.

    Variables calculated for each row (session audio):
    - Time spent speaking. Both the absolute value (s) and relative to the length of the recording (%).
    - Amount of speech, in terms of syllables spoken. Absolute value.
    - Speech rate, estimated from number of syllables.
    - Time spent in silence, inverse of time spent speaking, again, both absolute (s) and relative amount (%).

    :param df:           Pandas dataframe with columns "pitch", "srt_segment_info", and "speech_timeseries".
    :return: df_speech:  Same pandas dataframe as input, but with added vars / columns "speech_time_abs",
                         "speech_time_rel", "sil_time_abs", "sil_time_rel", "speech_syl", "speech_rate_syl".
    """

    # Deep copy input dataframe, so we do not break it if stg goes wrong.
    df_speech = pd.DataFrame(columns=df.columns, data=copy.deepcopy(df.values))

    # Init new vars / columns
    df_speech.loc[:, 'speech_time_abs'] = None
    df_speech.loc[:, 'speech_time_rel'] = None
    df_speech.loc[:, 'sil_time_abs'] = None
    df_speech.loc[:, 'sil_time_rel'] = None
    df_speech.loc[:, 'speech_syl'] = None
    df_speech.loc[:, 'speech_rate_syl'] = None

    # Loop through dataframe rows (session audios)
    for row_idx in df.index:

        # User feedback on progress
        if row_idx % 10 == 0:
            print('\nWorking on row ' + str(row_idx))

        # Reference necessary variables for ease of use.
        speech_ts = df_speech.loc[row_idx, 'speech_timeseries']  # speech_ts is binary
        srt_segments = df_speech.loc[row_idx, 'srt_segment_info']  # list of dicts

        # Get simple time-related measures.
        total_time_s = len(speech_ts) / TIMESERIES_SAMPLING_RATE_HZ
        df_speech.loc[row_idx, 'speech_time_abs'] = np.sum(speech_ts) / TIMESERIES_SAMPLING_RATE_HZ                 # s
        df_speech.loc[row_idx, 'speech_time_rel'] = df_speech.loc[row_idx, 'speech_time_abs'] / total_time_s * 100  # %
        df_speech.loc[row_idx, 'sil_time_abs'] = total_time_s - df_speech.loc[row_idx, 'speech_time_abs']           # s
        df_speech.loc[row_idx, 'sil_time_rel'] = 1 - df_speech.loc[row_idx, 'speech_time_rel']                      # %

        # Get total number of syllables
        speech_segments = [seg_dict['content'] for seg_dict in srt_segments]  # List of strings
        speech_syllables = get_syllables(speech_segments)[0]
        df_speech.loc[row_idx, 'speech_syl'] = speech_syllables                                                     # no
        df_speech.loc[row_idx, 'speech_rate_syl'] = speech_syllables / df_speech.loc[row_idx, 'speech_time_abs']    # syl/s

    return df_speech


def get_syllables(list_of_strings):
    """
    Helper function to extract the number of syllables from a list of strings.
    STRICTLY FOR HUNGARIAN, where syllables can be reliably estimated from the number of vowels.
    :param list_of_strings:   List of strings.
    :return: syllables_total: Integer, total number of syllables across list elements.
    :return: syllables:       List, with the number of syllables for each element in "list_of_strings" (integers).
    """
    vowels = [re.sub('[^aáeéiíoóöőüú]', '', s) for s in list_of_strings]
    syllables = [len(s) for s in vowels]
    syllables_total = sum(syllables)
    return syllables_total, syllables


def get_syllable_timeseries(df, timeseries_sr=TIMESERIES_SAMPLING_RATE_HZ):
    """
    Function to derive the syllable timeseries from the srt data of each segment, for each audio (row) in the
    input dataframe. For each audio (row), the resulting syllable timeseries has the same shape as the already existing
    binary speech timeseries.

    For a given speech segment, the sum of the syllable timeseries is the number of syllables spoken in that segment.
    Since we do not have word-level alignment yet, the number of syllables for a speech segment is divided by the
    number of samples in that segment, yielding the syllable timeseries values. E.g. for a segment of 500 samples and 10
    syllables, each sample has the value of 10 / 500 = 0.02. Samples outside of detected speech parts are set to 0.

    Outputs are a numpy array and a list of numpy arrays.

    :param df:                 Pandas dataframe with columns "pitch", "srt_segment_info", and "speech_timeseries".
    :param timeseries_sr:      Sampling rate of pitch timeseries. Should be the same value used for speech timeseries.
                               In order to ensure that, it defaults to module-level constant.
    :return: row_indices:      Numpy array of row indices, corresponding to dataframe rows used for arrays
                               in pitch_timeseries.
    :return: syl_timeseries:   List of numpy arrays, each one holding a syllable timeseries.
    """

    row_indices_list = []
    syl_timeseries = []

    for row_idx in df.index:

        # User feedback on progress
        if row_idx % 10 == 0:
            print('\nWorking on row ' + str(row_idx))

        # Data we need: list of pitch values, one array / segment, segment timing info, speech timeseries
        segment_times = df.loc[row_idx, 'srt_segment_info']
        speech_ts = df.loc[row_idx, 'speech_timeseries']

        # Transform segment timing info into lists of timestamps and sample numbers
        seg_starts = [s['start'] for s in segment_times]
        seg_ends = [s['end'] for s in segment_times]
        seg_starts_samples = [int(start_t * timeseries_sr) for start_t in seg_starts]
        seg_ends_samples = [int(end_t * timeseries_sr) for end_t in seg_ends]

        # Get segment content as a list of syllable numbers.
        seg_content = [s['content'] for s in segment_times]
        seg_syllables = get_syllables(seg_content)[1]

        # Init a numpy array of zeros with the same shape as speech timeseries. We will add the calculated pitch
        # segment arrays to this base array with the right indices.
        syl_ts = np.zeros(speech_ts.shape)

        # Loop through the segments and derive a syllable array for that segment, the same shape as
        # corresponding bit in the speech timeseries.
        for seg_idx in range(len(segment_times)):

            # Get the part of the speech timeseries corresponding to the current segment.
            seg_speech_ts = speech_ts[seg_starts_samples[seg_idx]: seg_ends_samples[seg_idx]]

            # Init an array with zeros with the same shape as the speech timeseries for this segment.
            seg_syl_ts = np.zeros(seg_speech_ts.shape)
            # Add the number of syllables in the current segment to each sample, and divide by the length.
            seg_syl_ts = (seg_syl_ts + seg_syllables[seg_idx]) / len(seg_syl_ts)

            # Add back the segment-level syllable array to the overall syllable timeseries array
            syl_ts[seg_starts_samples[seg_idx]: seg_ends_samples[seg_idx]] = seg_syl_ts

        # Store the row index and the final pitch timeseries in the output lists
        row_indices_list.append(row_idx)
        syl_timeseries.append(syl_ts)

    # Transform lists into numpy arrays before returning
    row_indices = np.asarray(row_indices_list)

    return row_indices, syl_timeseries



def get_pitch_timeseries(df, timeseries_sr=TIMESERIES_SAMPLING_RATE_HZ):
    """
    Function to derive the pitch timeseries from the pitch data of each segment, for each audio (row) in the
    input dataframe. For each audio (row), the resulting pitch timeseries has the same shape as the already existing
    binary speech timeseries.
    Outputs are a numpy array and a list of numpy arrays.

    :param df:                 Pandas dataframe with columns "pitch", "srt_segment_info", and "speech_timeseries".
    :param timeseries_sr:      Sampling rate of pitch timeseries. Should be the same value used for speech timeseries.
                               In order to ensure that, it defaults to module-level constant.
    :return: row_indices:      Numpy array of row indices, corresponding to dataframe rows used for arrays
                               in pitch_timeseries.
    :return: pitch_timeseries: List of numpy arrays, each one holding a pitch timeseries.
    """

    row_indices_list = []
    pitch_timeseries = []

    for row_idx in df.index:

        # User feedback on progress
        if row_idx % 10 == 0:
            print('\nWorking on row ' + str(row_idx))

        # Data we need: list of pitch values, one array / segment, segment timing info, speech timeseries
        pitch_list = df.loc[row_idx, 'pitch']
        segment_times = df.loc[row_idx, 'srt_segment_info']
        speech_ts = df.loc[row_idx, 'speech_timeseries']

        # Transform segment timing info into lists of timestamps and sample numbers
        seg_starts = [s['start'] for s in segment_times]
        seg_ends = [s['end'] for s in segment_times]
        seg_starts_samples = [int(start_t * timeseries_sr) for start_t in seg_starts]
        seg_ends_samples = [int(end_t * timeseries_sr) for end_t in seg_ends]

        # Init a numpy array of zeros with the same shape as speech timeseries. We will add the calculated pitch
        # segment arrays to this base array with the right indices.
        pitch_ts = np.zeros(speech_ts.shape)

        # Loop through the segments and upsample / interpolate the pitch array for that segment the same shape as
        # corresponding bit in the speech timeseries.
        for seg_idx in range(len(segment_times)):

            # Get the part of the speech timeseries corresponding to the current segment
            seg_speech_ts = speech_ts[seg_starts_samples[seg_idx]: seg_ends_samples[seg_idx]]

            # Init an array with zeros with the same shape as the speech timeseries for this segment
            seg_pitch_ts = np.zeros(seg_speech_ts.shape)

            # Upsample / interpolate pitch values into this segment-shaped array
            pitch_array = pitch_list[seg_idx]
            # Only upsample existing (not empty) pitch array, otherwise the seg_int_ts var just remains an array
            # of zeros.
            if list(pitch_array):
                xp = np.arange(0, len(pitch_array), 1)
                x = np.arange(0, len(pitch_array)-1, (len(pitch_array)-1) / (len(seg_pitch_ts)-1))
                if (len(x) + 1) == len(seg_pitch_ts):
                    x = np.concatenate((x, np.asarray([xp[-1]])))
                seg_pitch_ts[0:] = np.interp(x, xp, pitch_array)

            # Sanity check about shape
            assert seg_pitch_ts.shape == seg_speech_ts.shape, 'Wrong segment pitch array shape!!!'

            # Add back the interpolated segment-level pitch array to the overall pitch timeseries array
            pitch_ts[seg_starts_samples[seg_idx]: seg_ends_samples[seg_idx]] = seg_pitch_ts

        # Store the row index and the final pitch timeseries in the output lists
        row_indices_list.append(row_idx)
        pitch_timeseries.append(pitch_ts)

    # Transform lists into numpy arrays before returning
    row_indices = np.asarray(row_indices_list)

    return row_indices, pitch_timeseries


def get_intensity_timeseries(df, timeseries_sr=TIMESERIES_SAMPLING_RATE_HZ):
    """
    Function to derive the intensity timeseries from the intensity data of each segment, for each audio (row) in the
    input dataframe. For each audio (row), the resulting intensity timeseries has the same shape as the already existing
    binary speech timeseries.
    Outputs are a numpy array and a list of numpy arrays.

    :param df:              Pandas dataframe with columns "intensity", "srt_segment_info", and "speech_timeseries".
    :param timeseries_sr:   Sampling rate of intensity timeseries. Should be the same value used for speech timeseries.
                            In order to ensure that, it defaults to module-level constant.
    :return: row_indices:          Numpy array of row indices, corresponding to dataframe rows used for arrays
                                   in intensity_timeseries.
    :return: intensity_timeseries: List of numpy arrays, each one holding an intensity timeseries.
    """

    row_indices_list = []
    intensity_timeseries = []

    for row_idx in df.index:

        # User feedback on progress
        if row_idx % 10 == 0:
            print('\nWorking on row ' + str(row_idx))

        # Data we need: list of intensity values, one array / segment, segment timing info, speech timeseries
        intensity_list = df.loc[row_idx, 'intensity']
        segment_times = df.loc[row_idx, 'srt_segment_info']
        speech_ts = df.loc[row_idx, 'speech_timeseries']

        # Transform segment timing info into lists of timestamps and sample numbers
        seg_starts = [s['start'] for s in segment_times]
        seg_ends = [s['end'] for s in segment_times]
        seg_starts_samples = [int(start_t * timeseries_sr) for start_t in seg_starts]
        seg_ends_samples = [int(end_t * timeseries_sr) for end_t in seg_ends]

        # Init a numpy array of zeros with the same shape as speech timeseries. We will add the calculated intensity
        # segment arrays to this base array with the right indices.
        intensity_ts = np.zeros(speech_ts.shape)

        # Loop through the segments and upsample / interpolate the intensity array for that segment the same shape as
        # corresponding bit in the speech timeseries.
        for seg_idx in range(len(segment_times)):

            # Get the part of the speech timeseries corresponding to the current segment
            seg_speech_ts = speech_ts[seg_starts_samples[seg_idx]: seg_ends_samples[seg_idx]]

            # Init an array with zeros with the same shape as the speech timeseries for this segment
            seg_int_ts = np.zeros(seg_speech_ts.shape)

            # Upsample / interpolate intensity values into this segment-shaped array
            int_array = intensity_list[seg_idx]
            # Only upsample existing (not empty) intensity array, otherwise the seg_int_ts var just remains an array
            # of zeros.
            if list(int_array):
                xp = np.arange(0, len(int_array), 1)
                x = np.arange(0, len(int_array)-1, (len(int_array)-1) / (len(seg_int_ts)-1))
                if (len(x) + 1) == len(seg_int_ts):
                    x = np.concatenate((x, np.asarray([xp[-1]])))
                seg_int_ts[0:] = np.interp(x, xp, int_array)

            # Sanity check about shape
            assert seg_int_ts.shape == seg_speech_ts.shape, 'Wrong segment intensity array shape!!!'

            # Add back the interpolated segment-level intensity array to the overall intensity timeseries array
            intensity_ts[seg_starts_samples[seg_idx]: seg_ends_samples[seg_idx]] = seg_int_ts

        # subtract lowest positive (nonzero) value
        min_intensity = np.min(intensity_ts[np.nonzero(intensity_ts)])
        intensity_ts[np.nonzero(intensity_ts)] = intensity_ts[np.nonzero(intensity_ts)] - min_intensity

        # Store the row index and the final intensity timeseries in the output lists
        row_indices_list.append(row_idx)
        intensity_timeseries.append(intensity_ts)

    # Transform lists into numpy arrays before returning
    row_indices = np.asarray(row_indices_list)

    return row_indices, intensity_timeseries


def add_pitch_intensity(df):
    """
    Function to call praat-parselmouth intensity and pitch estimation methods on all audio segments the dataframe
    refers to. Input dataframe must have columns "segments_dir", "segments", and "srt_segment_info".
    Output is the same dataframe but with added columns "intensity" and "pitch". They both hold a list in each row,
    with list elements being the intensity / pitch values in numpy arrays.

    Important parameters are defined as module constants, better not mess with them.

    :param df:              Pandas dataframe with each row holding variables for one audio session. Must have columns
                            "segments_dir", "segments", and "srt_segment_info".
    :return: df_pitch_int:  Same as input dataframe but with two new columns: "intensity" and "pitch".
    """

    # Deep copy input dataframe, so we do not break it if stg goes wrong.
    df_pitch_int = pd.DataFrame(columns=df.columns, data=copy.deepcopy(df.values))

    df_pitch_int.loc[:, 'intensity'] = None
    df_pitch_int.loc[:, 'pitch'] = None

    # Loop through rows (audio files) of df.
    for row_idx in df_pitch_int.index:

        # User feedback on progress.
        if row_idx % 10 == 0:
            print('Working on row ' + str(row_idx))

        # Relevant vars from dataframe.
        segments_dir = df_pitch_int.loc[row_idx, 'segments_dir']
        segments = df_pitch_int.loc[row_idx, 'segments']
        # Init lists for intensity and pitch arrays.
        intensity_list = []
        pitch_list = []

        # Loop through segment audios
        for seg_idx, seg in enumerate(segments):

            seg_timing = df_pitch_int.loc[row_idx, 'srt_segment_info'][seg_idx]

            # There is a minimum required length for calculating pitch or intensity on an audio segment.
            if seg_timing['end'] - seg_timing['start'] >= MIN_SEGMENT_LENGTH_S:
                # Load audio segment with parselmouth.
                segment_path = os.path.join(segments_dir, seg)
                sound = pm.Sound(segment_path)
                # The parselmouth "sound" object has methods for intensity and pitch estimation.
                intensity = sound.to_intensity(time_step=PM_TIME_STEP, minimum_pitch=PM_MINIMUM_PITCH_HZ,
                                               subtract_mean=False)
                pitch = sound.to_pitch_ac(time_step=PM_TIME_STEP, pitch_floor=PM_MINIMUM_PITCH_HZ,
                                          pitch_ceiling=PM_MAXIMUM_PITCH_HZ, very_accurate=PM_PITCH_ACCURATE,
                                          voicing_threshold=PM_VOICING_THRESHOLD)
                # Transform intensity and pitch objects into numpy arrays.
                int_array = np.squeeze(intensity.as_array())
                pitch_array = pitch.to_array()  # still contains all pitch candidates
                pitch_array = np.squeeze(pitch_array[0, :])  # strip to first (strongest) candidate, array of tuples
                pitch_array = np.asarray([i[0] for i in pitch_array])  # get only the pitch values, not their strength
                # Finally, append intensity and pitch lists with the numpy arrays.
                intensity_list.append(int_array)
                pitch_list.append(pitch_array)

            # If the audio segment was not long enough for intensity / pitch estimation,
            # use an empty list instead of numpy array.
            else:
                intensity_list.append([])
                pitch_list.append([])

        # Store intensity and pitch lists for the current row in the dataframe.
        df_pitch_int.at[row_idx, 'intensity'] = intensity_list
        df_pitch_int.at[row_idx, 'pitch'] = pitch_list

    return df_pitch_int


def main():
    # Input argument handling. One positional argument for the folder path to relevant .pkl / .npz files.
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='Path to directory holding transcription- and prosody-related pickle (.pkl) and '
                             'potentially numpy (.npz) files.')
    args = parser.parse_args()

    # Load pandas dataframe with all necessary variables, should be the final output from "vad_on_segments.py".
    init_pkl_path = os.path.join(args.data_dir, INPUT_PICKLE_FILENAME)
    df_init = pd.read_pickle(init_pkl_path)

    # Add pitch and intensity data to dataframe, save it out.
    df_pitch_intensity = add_pitch_intensity(df_init)
    pkl_with_pitch_int_path = os.path.join(args.data_dir, PITCH_INTENSITY_PICKLE_FILENAME)
    df_pitch_intensity.to_pickle(pkl_with_pitch_int_path)

    # Get intensity timeseries, save out resulting variables with pickle.
    intensity_row_indices, intensity_timeseries = get_intensity_timeseries(df_pitch_intensity)
    intensity_pkl_path = os.path.join(args.data_dir, INTENSITY_PKL_FILENAME)
    with open(intensity_pkl_path, 'wb') as pkl_out:
        pickle.dump([intensity_row_indices, intensity_timeseries], pkl_out)

    # Get pitch timeseries, save out resulting variables with pickle.
    pitch_row_indices, pitch_timeseries = get_pitch_timeseries(df_pitch_intensity)
    pitch_pkl_path = os.path.join(args.data_dir, PITCH_PKL_FILENAME)
    with open(pitch_pkl_path, 'wb') as pkl_out:
        pickle.dump([pitch_row_indices, pitch_timeseries], pkl_out)

    # Get syllable timeseries, save out resulting variables with pickle.
    syl_row_indices, syl_timeseries = get_syllable_timeseries(df_pitch_intensity)
    syl_pkl_path = os.path.join(args.data_dir, SYLLABLE_PKL_FILENAME)
    with open(syl_pkl_path, 'wb') as syl_out:
        pickle.dump([syl_row_indices, syl_timeseries], syl_out)

    # Get speech related aggregate variables for each row in dataframe.
    df_speech = add_speech_vars_to_df(df_pitch_intensity)
    pkl_with_speech_path = os.path.join(args.data_dir, PITCH_INTENSITY_SPEECHVARS_PICKLE_FILENAME)
    df_speech.to_pickle(pkl_with_speech_path)


if __name__ == '__main__':
    main()
