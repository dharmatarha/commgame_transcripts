"""
Module / script to add new speech-turn - based variables as potential predictors to the prosody stat tables.

Vars to derive:
- Speech turn length: mean and sd of length for individual, mean and diff for pair, correlations
- Response time: mean and sd of response time for individual, mean and diff for pair, correlations
- Pauses: silent parts within someone's speech turn, frequency and length per time (per minute spent speaking and
per overall speech time), mean and diff for pair, (windowed correlation?)
- Backchannels: frequency of backchannels / minute or / other's turn, mean and diff for pair, (windowed correlation?)
- Interruptions: "Taking over" of speech turns, frequency of interruptions / minute or / other's turn,
mean and diff for pair, (windowed correlation?)

There are two types of correlations calculated for participants:
(1) Correlations across paired-up speech turn attributes of participants, with both temporal directions (+1 / -1 lag
cross-correlations). To pair up turns, speech turns are sorted according to their start times, with backchannels and
very short speech turns disregarded.
(2) First-order auto-correlation of speech turn attributes.

Params:


"""


import pandas as pd
import numpy as np
import os
import pickle
import copy
import srt
import datetime
import sys

sys.path.append('/home/adamb/commgame_transcripts/postproc/')

from srt_to_text import replace_patterns_in_subs
from srt_to_text import filter_tags_in_subs
from srt_to_text import clear_empty_subs
from srt_to_text import merge_close_subs
from srt_to_text import norm_whitespaces_in_subs


DATA_DIR = '/media/adamb/data_disk/prosody_data/'
DF_PKL = 'prosody_w_speech_int_pitch_df.pkl'
BASE_TABLE_CSV = 'base_data_form.csv'
# Known cases of missing data
MISSING_AUDIO_DATA = [('pair100', 'BG4', 'Mordor'),
                      ('pair100', 'BG4', 'Gondor'),
                      ('pair188', 'BG5', 'Mordor'),
                      ('pair188', 'BG5', 'Gondor')]
BASE_TABLE_CSV_OUT = 'new_speech_vars.csv'
TIMESERIES_SAMPLING_RATE_HZ = 200
RESAMPLING_RATE = 16000
REPLACEMENTS = {
    'erazmusz': 'Erasmus',
    'eraszmusz': 'Erasmus',
    'erasmus': 'Erasmus',
    'erazmus': 'Erasmus',
    'erazmuz': 'Erasmus',
    'Erazmusz': 'Erasmus',
    'Eraszmusz': 'Erasmus',
    'Erazmus': 'Erasmus',
    'Erazmuz': 'Erasmus',
    'elte': 'ELTE',
    'Elte': 'ELTE',
    'eltés': 'ELTE-s',
    'bme': 'BME',
    'Bme': 'BME',
    'bmés': 'BME-s',
    'károli': 'Károli',
    'Károlyi': 'Károli',
    '<laugh<': '<laugh>',
    '>laugh>': '<laugh>',
    '>laugh<': '<laugh>',
    '<hes<': '<hes>',
    '>hes>': '<hes>',
    '>hes<': '<hes>',
    '<hum<': '<hum>',
    '>hum>': '<hum>',
    '>hum<': '<hum>',
    'Semmelweisz': 'Semmelweis',
    'Semmelweiss': 'Semmelweis'}
SENTENCE_ENDING_PUNCTUATION_MARKS = ['.', '?', '!']
DEFAULT_SPEAKER_TAGS = ('Mordor: ', 'Gondor: ')
TURN_MERGE_THR = 2


def subs_to_srt_list(subs_m, subs_g):
    subs_list_m = copy.deepcopy(subs_m)
    subs_list_g = copy.deepcopy(subs_g)
    # Get lists of srt objects from subtitle segments (dicts)
    srt_list_m = []
    for sub_idx, sub in enumerate(subs_list_m):
        srt_obj = srt.Subtitle(index=sub_idx,
                               start=datetime.timedelta(seconds=sub['start']),
                               end=datetime.timedelta(seconds=sub['end']),
                               content=sub['content'])
        srt_list_m.append(srt_obj)
    srt_list_g = []
    for sub_idx, sub in enumerate(subs_list_g):
        srt_obj = srt.Subtitle(index=sub_idx,
                               start=datetime.timedelta(seconds=sub['start']),
                               end=datetime.timedelta(seconds=sub['end']),
                               content=sub['content'])
        srt_list_g.append(srt_obj)
    # Add speaker tags to each srt content
    for sub in srt_list_m:
        sub.content = ' '.join([DEFAULT_SPEAKER_TAGS[0], sub.content])
    for sub in srt_list_g:
        sub.content = ' '.join([DEFAULT_SPEAKER_TAGS[1], sub.content])
    # Concatenate lists and sort them according to start time.
    joint_srt_list = srt_list_m + srt_list_g
    joint_srt_list.sort(key=lambda x: x.start)

    return joint_srt_list


def srt_per_speaker(joint_subs_list):
    subs_list = copy.deepcopy(joint_subs_list)
    # Divide joint list according to speakers, into two
    srt_list_m = [sub for sub in subs_list if sub.content.startswith(DEFAULT_SPEAKER_TAGS[0])]
    srt_list_g = [sub for sub in subs_list if sub.content.startswith(DEFAULT_SPEAKER_TAGS[1])]
    srt_idx_m = [idx for idx, sub in enumerate(subs_list) if sub.content.startswith(DEFAULT_SPEAKER_TAGS[0])]
    srt_idx_g = [idx for idx, sub in enumerate(subs_list) if sub.content.startswith(DEFAULT_SPEAKER_TAGS[1])]
    return srt_list_m, srt_list_g, srt_idx_m, srt_idx_g


def prepare_subs(subs_m, subs_g):

    joint_srt_list = subs_to_srt_list(subs_m, subs_g)
    joint_srt_list = replace_patterns_in_subs(joint_srt_list)
    joint_srt_list = filter_tags_in_subs(joint_srt_list)
    joint_srt_list = clear_empty_subs(joint_srt_list)
    # Merge is called multiple times as the function only handles non-recursive.
    joint_srt_list = merge_close_subs(joint_srt_list,
                                      speaker_tags=DEFAULT_SPEAKER_TAGS,
                                      time_thr=TURN_MERGE_THR,
                                      punct_check=False)
    joint_srt_list = merge_close_subs(joint_srt_list,
                                      speaker_tags=DEFAULT_SPEAKER_TAGS,
                                      time_thr=TURN_MERGE_THR,
                                      punct_check=False)
    joint_srt_list = norm_whitespaces_in_subs(joint_srt_list)

    return joint_srt_list


def get_turn_length_stats(srt_obj_list):
    subs_list = copy.deepcopy(srt_obj_list)
    subs_len = np.asarray([sub.end.total_seconds() - sub.start.total_seconds() for sub in subs_list])
    if any(subs_len < 0):
        raise ValueError('Negative speech turn length!')
    return np.mean(subs_len), np.median(subs_len), np.std(subs_len), subs_len


def get_backchannels(srt_obj_list, max_length_s=1.0, padding_s=0.3):
    subs_list = copy.deepcopy(srt_obj_list)  # Avoid messing up input arg list in place
    # Sanity check, is the subtitles list sorted?
    sorted_bool = all([srt_obj_list[i+1].start >= srt_obj_list[i].start for i in range(len(srt_obj_list)-1)])
    if not sorted_bool:
        raise AssertionError('Input arg "subtitles_list" MUST be sorted according to subtitle start times!')
    # Init lists for holding backchannel srt objects and their indices in initial srt list.
    backchannels = []
    backchannels_idx = []
    # Loop through subtitles, always compare current with next one
    for sub_idx, sub in enumerate(subs_list):
        # If we are not at the end of the subtitle list, compare the timing of the current and the next subtitle.
        if sub_idx != len(subs_list)-1:
            sub_next = subs_list[sub_idx + 1]
            # If the next subtitle happens during the time of the current subtitle (considering padding as well),
            # and it is considered short enough, delete it.
            if (sub_next.end.total_seconds() + padding_s <= sub.end.total_seconds()) and \
                    (sub_next.start.total_seconds()-padding_s >= sub.start.total_seconds()) and \
                    (sub_next.end-sub_next.start).total_seconds() <= max_length_s:
                backchannels.append(sub_next)
                backchannels_idx.append(sub_idx + 1)

    return backchannels, backchannels_idx


def get_resp_times(srt_obj_list, set_negative_to_zero=True):
    subs_list = copy.deepcopy(srt_obj_list)  # Avoid messing up input arg list in place
    # Sanity check, is the subtitles list sorted?
    sorted_bool = all([srt_obj_list[i+1].start >= srt_obj_list[i].start for i in range(len(srt_obj_list)-1)])
    if not sorted_bool:
        raise AssertionError('Input arg "subtitles_list" MUST be sorted according to subtitle start times!')
    # Init result lists.
    resp_times_list = []
    resp_times_indices = []
    resp_t_m = []
    resp_t_g = []
    resp_idx_m = []
    resp_idx_g = []
    # Loop through srt objects, check if the current and next turns belong to different speakers, and get response
    # time if yes. Collect indices of turns (srt objects) that the response times belong to.from
    for sub_current_idx, sub_current in enumerate(subs_list):
        # If we are not at the end of the subtitle list, compare the timing of the current and the next subtitle.
        if sub_current_idx < len(subs_list)-1:
            sub_next = subs_list[sub_current_idx + 1]
            # Check if speakers are different
            current_speaker = [tag for tag in DEFAULT_SPEAKER_TAGS if sub_current.content.startswith(tag)]
            next_speaker = [tag for tag in DEFAULT_SPEAKER_TAGS if sub_next.content.startswith(tag)]
            # Sanity check: could we identify the speakers?
            assert bool(current_speaker) and bool(next_speaker), 'Could not determine speaker!'
            # Check for equality of speakers, get response time if they are different.
            if current_speaker[0] != next_speaker[0]:
                resp_time = sub_next.start.total_seconds() - sub_current.end.total_seconds()
                resp_times_list.append(resp_time)
                resp_times_indices.append(sub_current_idx + 1)
                # Sort into M - G result lists
                if next_speaker[0] == DEFAULT_SPEAKER_TAGS[0]:
                    resp_t_m.append(resp_time)
                    resp_idx_m.append(sub_current_idx + 1)
                elif next_speaker[0] == DEFAULT_SPEAKER_TAGS[1]:
                    resp_t_g.append(resp_time)
                    resp_idx_g.append(sub_current_idx + 1)
                else:
                    raise ValueError('Speaker is not in the first two values of DEFAULT_SPEAKER_TAGS!')

    # Depending on flag, set negative response time values to zero.
    if set_negative_to_zero:
        resp_times_list = [0 if t < 0 else t for t in resp_times_list if t < 0]
        resp_t_m = [0 if t < 0 else t for t in resp_t_m if t < 0]
        resp_t_g = [0 if t < 0 else t for t in resp_t_g if t < 0]

    return np.asarray(resp_times_list), np.asarray(resp_times_indices), np.asarray(resp_t_m),\
        np.asarray(resp_idx_m), np.asarray(resp_t_g), np.asarray(resp_idx_g)


def get_interrupts(srt_obj_list, interrupt_start_min=0.5, interrupt_end_min=0.5, interrupt_len_plus=2):
    subs_list = copy.deepcopy(srt_obj_list)  # Avoid messing up input arg list in place
    # Sanity check, is the subtitles list sorted?
    sorted_bool = all([srt_obj_list[i+1].start >= srt_obj_list[i].start for i in range(len(srt_obj_list)-1)])
    if not sorted_bool:
        raise AssertionError('Input arg "subtitles_list" MUST be sorted according to subtitle start times!')
    # Init result lists.
    interrupt_indices = []
    interrupt_idx_m = []
    interrupt_idx_g = []
    # Loop through srt objects, check if the current and next turns belong to different speakers, and check
    # subtitle timings if yes.
    for sub_current_idx, sub_current in enumerate(subs_list):
        # If we are not at the end of the subtitle list, compare the timing of the current and the next subtitle.
        if sub_current_idx != len(subs_list)-1:
            sub_next = subs_list[sub_current_idx + 1]
            # Check if speakers are different
            current_speaker = [tag for tag in DEFAULT_SPEAKER_TAGS if sub_current.content.startswith(tag)][0]
            next_speaker = [tag for tag in DEFAULT_SPEAKER_TAGS if sub_next.content.startswith(tag)][0]
            # Sanity check: could we identify the speakers?
            assert bool(current_speaker) and bool(next_speaker), 'Could not determine speaker!'
            # Check for equality of speakers.
            if current_speaker != next_speaker:
                # Check for subtitle overlap.
                if sub_next.start.total_seconds() >= sub_current.start.total_seconds() + interrupt_start_min and \
                        sub_next.start.total_seconds() <= sub_current.end.total_seconds() - interrupt_end_min and \
                        sub_next.end.total_seconds() >= sub_current.end.total_seconds() + interrupt_len_plus:
                    interrupt_indices.append(sub_current_idx + 1)
                    # Sort into M - G result lists
                    if next_speaker == DEFAULT_SPEAKER_TAGS[0]:
                        interrupt_idx_m.append(sub_current_idx + 1)
                    elif next_speaker == DEFAULT_SPEAKER_TAGS[1]:
                        interrupt_idx_g.append(sub_current_idx + 1)
                    else:
                        raise ValueError('Speaker is not in the first two values of DEFAULT_SPEAKER_TAGS!')

    return interrupt_indices, interrupt_idx_m, interrupt_idx_g


def get_srt_cross_corrs(values_m, values_g, indices_m, indices_g):
    # Get corresponding pairs of values, first for Mordor.
    paired_idx_m = []
    paired_idx_g = []
    previous_idx_g = 0
    for idx_m in indices_m:
        mask_g = np.where(indices_g < idx_m)[0]
        if mask_g.size != 0:
            idx_g = np.max(indices_g[mask_g])
            if idx_g != previous_idx_g:
                previous_idx_g = idx_g
                paired_idx_m.append(np.where(indices_m == idx_m)[0][0])
                paired_idx_g.append(np.where(indices_g == idx_g)[0][0])
    paired_idx_m = np.asarray(paired_idx_m)
    paired_idx_g = np.asarray(paired_idx_g)
    corr_m = np.corrcoef(values_m[paired_idx_m], values_g[paired_idx_g])[0, 1]
    # Gondor.
    paired_idx_m = []
    paired_idx_g = []
    previous_idx_m = 0
    for idx_g in indices_g:
        mask_m = np.where(indices_m < idx_g)[0]
        if mask_m.size != 0:
            idx_m = np.max(indices_m[mask_m])
            if idx_m != previous_idx_m:
                previous_idx_m = idx_m
                paired_idx_g.append(np.where(indices_g == idx_g)[0][0])
                paired_idx_m.append(np.where(indices_m == idx_m)[0][0])
    paired_idx_g = np.asarray(paired_idx_g)
    paired_idx_m = np.asarray(paired_idx_m)
    corr_g = np.corrcoef(values_m[paired_idx_m], values_g[paired_idx_g])[0, 1]

    return corr_m, corr_g


def get_pauses_in_speech_turns(speech_timeseries, srt_obj_list, pause_min_len_s=0.2,
                               speech_min_len_s=0.2, sampling_rate=TIMESERIES_SAMPLING_RATE_HZ):
    """
    Function to calculate the sum of pauses in speech segments. Pauses are defined as silent parts within a speech turn
    with at least "pause_min_len_s" length (sane values are ~0.18 - 0.2 seconds).
    :param speech_timeseries: Numpy array, binary, 1D. Voice detection array, with ones corresponding to speech,
                              zeros to silence. The array corresponds to the same audio recording that "srt_obj_list"
                              holds the subtitles / transcription for. Its sampling rate is provided in "sampling_rate".
    :param srt_obj_list:      List of srt subtitle objects, corresponding to speech turns detected in same audio
                              recording that "speech_timeseries" describes as well.
    :param pause_min_len_s:   Numeric value, minimum length of silent segment to be considered a pause, in seconds.
                              Defaults to 0.2.
    :param speech_min_len_s:  Numeric value, minimum length of speech segment before and after silences in order for
                              them to be considered pauses. It is in seconds. Calculated cumulatively. Defaults to 0.2.
                              E.g. for a speech (sp) and silences (si) segment with the following timings (in seconds):
                              sp(0.5) - si(0.4) - sp(0.15) - si(0.5) - sp(1.1), both silent parts are considered pauses
                              as there have been enough speech before and after them cumulatively, even if the middle
                              speech segment (sp(0.15)) is too short to fulfill this condition by its own.
    :param sampling_rate:     Numeric value, sampling rate of "speech_timeseries" in Hz. Defaults to module-level
                              constant TIMESERIES_SAMPLING_RATE_HZ.
    :return: pause_sum:       Numpy array, 1D, where each value corresponds to the total amount of pause in a speech
                              turn. The length of the array is the same as the length of "srt_obj_list". Values are in
                              seconds.
    """
    sp_ts = copy.deepcopy(speech_timeseries)  # Avoid messing up input arg list in place
    subs_list = copy.deepcopy(srt_obj_list)  # Avoid messing up input arg list in place
    # Pauses are silent parts longer than pause_min_len_s.
    pause_no = []
    pause_sum = []
    speech_sum = []
    for sub in subs_list:
        ts = sp_ts[round(sub.start.total_seconds() * sampling_rate):
                   round(sub.end.total_seconds() * sampling_rate)]
        zero_ranges = zero_runs(ts)
        one_ranges = zero_runs((ts - 1) * (-1))
        zero_len_s = (zero_ranges[:, 1] - zero_ranges[:, 0]) / sampling_rate
        one_len_s = (one_ranges[:, 1] - one_ranges[:, 0]) / sampling_rate
        total_speech_s = np.sum(one_len_s)
        # There are only any silences to consider if there are at least 2 runs of ones (2 speech parts) in the segment,
        # and at least one zero run (silent part). Otherwise there is none.
        if zero_ranges.shape[0] > 0 and one_ranges.shape[0] > 1:
            # Pause can only be a zero run (silent part) between runs of ones (speech parts). A further condition is
            # that the runes of ones (speech parts) before and after the silence must be at least speech_min_len_s long.
            # Important: speech length is calculated cumulatively so that we do not throw away silence because of a
            # short, intermittent speech burst at its end, later followed by a longer speech segment.
            # Boolean array for marking which zero run can be considered a pause:
            zero_run_is_pause = np.zeros(zero_len_s.shape).astype(bool)
            for row_idx in range(zero_ranges.shape[0]):
                # Flags for tracking the necessary conditions for declaring a silent part a pause.
                speech_before_flag = False
                speech_after_flag = False
                # Get the number of rows in one_ranges before current zero run and check their cumulative length.
                ones_before_current_silence_idx = np.where(one_ranges[:, 0] < zero_ranges[row_idx, 0])[0]  # Returns an array.
                if ones_before_current_silence_idx.size > 0 and \
                        np.sum(one_len_s[ones_before_current_silence_idx]) > speech_min_len_s:
                    speech_before_flag = True
                # Get the number of rows in one_ranges following current zero run and check their cumulative length.
                ones_after_current_silence_idx = np.where(one_ranges[:, 0] > zero_ranges[row_idx, 0])[0]  # Returns an array.
                if ones_after_current_silence_idx.size > 0 and \
                        np.sum(one_len_s[ones_after_current_silence_idx]) > speech_min_len_s:
                    speech_after_flag = True
                # Check if all conditions are met.
                if speech_before_flag and speech_after_flag and zero_len_s[row_idx] >= pause_min_len_s:
                    zero_run_is_pause[row_idx] = True
            # After looping through the zero runs, the total pause is the sum of pauses.
            pause_sum.append(np.sum(zero_len_s[zero_run_is_pause]))
            # Number of pauses is just the no. of zero runs meeting all conditions.
            pause_no.append(np.sum(zero_run_is_pause))

        else:
            pause_sum.append(0)
            pause_no.append(0)

        speech_sum.append(total_speech_s)

    return np.asarray(pause_sum), np.asarray(pause_no), np.asarray(speech_sum)


def zero_runs(arr):
    """
    Cool utility that finds the zero runs in the input signal.
    :param arr: Numpy array, 1-dimensional.
    :return:    Numpy array, with shape (zero_runs, 2), with each row containing the range of a
                zero run (start and end).
    """
    # Create an array that is 1 where arr is 0, and pad each end with an extra 0.
    is_zero = np.concatenate(([0], np.equal(arr, 0), [0]))
    abs_diff = np.abs(np.diff(is_zero))
    # Runs start and end where abs_diff is 1.
    ranges = np.where(abs_diff == 1)[0].reshape(-1, 2)
    return ranges


def main():
    # Load df with prosody vars
    pros_df = pd.read_pickle(os.path.join(DATA_DIR, DF_PKL))
    # vars about speech turn length
    pros_df.loc[:, 'mean_turn_length'] = None
    pros_df.loc[:, 'std_turn_length'] = None
    pros_df.loc[:, 'coeff_of_var_turn_length'] = None
    pros_df.loc[:, 'autocorr_turn_length'] = None
    pros_df.loc[:, 'crosscorr_turn_length'] = None
    # vars about response times (speech turn response times)
    pros_df.loc[:, 'mean_resp_time'] = None
    pros_df.loc[:, 'std_resp_time'] = None
    pros_df.loc[:, 'coeff_of_var_resp_time'] = None
    pros_df.loc[:, 'autocorr_resp_time'] = None
    pros_df.loc[:, 'crosscorr_resp_time'] = None
    # vars about backchannels
    pros_df.loc[:, 'backch_no'] = None
    pros_df.loc[:, 'backch_rate'] = None
    # vars about interruptions
    pros_df.loc[:, 'interrupt_no'] = None
    pros_df.loc[:, 'interrupt_rate'] = None
    # vars about number of turns
    pros_df.loc[:, 'turn_no'] = None
    pros_df.loc[:, 'turn_rate'] = None
    # vars about pauses
    pros_df.loc[:, 'pause_no'] = None
    pros_df.loc[:, 'pause_sum'] = None
    pros_df.loc[:, 'pause_ratio'] = None
    pros_df.loc[:, 'pause_per_speech_min'] = None
    pros_df.loc[:, 'pause_sum_per_turn'] = None

    # Load csv
    stat_df = pd.read_csv(os.path.join(DATA_DIR, BASE_TABLE_CSV))
    # Add new vars / columns
    # vars about speech turn length
    stat_df.loc[:, 'mean_turn_length'] = None
    stat_df.loc[:, 'diff_turn_length'] = None
    stat_df.loc[:, 'mean_std_turn_length'] = None
    stat_df.loc[:, 'diff_std_turn_length'] = None
    stat_df.loc[:, 'mean_coeff_var_turn_length'] = None
    stat_df.loc[:, 'diff_coeff_var_turn_length'] = None
    stat_df.loc[:, 'mean_autocorr_turn_length'] = None
    stat_df.loc[:, 'diff_autocorr_turn_length'] = None
    stat_df.loc[:, 'mean_crosscorr_turn_length'] = None
    stat_df.loc[:, 'diff_crosscorr_turn_length'] = None
    # vars about response times (speech turn response times)
    stat_df.loc[:, 'mean_resp_time'] = None
    stat_df.loc[:, 'diff_resp_time'] = None
    stat_df.loc[:, 'mean_std_resp_time'] = None
    stat_df.loc[:, 'diff_std_resp_time'] = None
    stat_df.loc[:, 'mean_coeff_var_resp_time'] = None
    stat_df.loc[:, 'diff_coeff_var_resp_time'] = None
    stat_df.loc[:, 'mean_autocorr_resp_time'] = None
    stat_df.loc[:, 'diff_autocorr_resp_time'] = None
    stat_df.loc[:, 'mean_crosscorr_resp_time'] = None
    stat_df.loc[:, 'diff_crosscorr_resp_time'] = None
    # vars about backchannels
    stat_df.loc[:, 'mean_backch_no'] = None
    stat_df.loc[:, 'diff_backch_no'] = None
    stat_df.loc[:, 'mean_backch_rate'] = None
    stat_df.loc[:, 'diff_backch_rate'] = None
    # vars about interruptions
    stat_df.loc[:, 'mean_interrupt_no'] = None
    stat_df.loc[:, 'diff_interrupt_no'] = None
    stat_df.loc[:, 'mean_interrupt_rate'] = None
    stat_df.loc[:, 'diff_interrupt_rate'] = None
    # vars about turn numbers
    stat_df.loc[:, 'mean_turn_no'] = None
    stat_df.loc[:, 'diff_turn_no'] = None
    stat_df.loc[:, 'mean_turn_rate'] = None
    stat_df.loc[:, 'diff_turn_rate'] = None
    # vars about pauses
    stat_df.loc[:, 'mean_pause_no'] = None
    stat_df.loc[:, 'diff_pause_no'] = None
    stat_df.loc[:, 'mean_pause_sum'] = None
    stat_df.loc[:, 'diff_pause_sum'] = None
    stat_df.loc[:, 'mean_pause_sum_per_min'] = None
    stat_df.loc[:, 'diff_pause_sum_per_min'] = None
    stat_df.loc[:, 'mean_pause_sum_per_turn'] = None
    stat_df.loc[:, 'diff_pause_sum_per_turn'] = None

    # Loop through real and pseudo pairs (rows) of stat_df
    for pair_idx in stat_df.index:

        # User feedback on progress.
        if pair_idx % 100 == 0:
            print('At pair ' + str(pair_idx))

        # Identifiers of pair necessary for selecting right rows of pros_df.
        pair_m = stat_df.loc[pair_idx, 'Mordor_pairNo']
        pair_g = stat_df.loc[pair_idx, 'Gondor_pairNo']
        session = stat_df.loc[pair_idx, 'BG_label']

        # Check if current pair has missing audio data (so we have to jump over them)
        subject_m_tuple = ('pair' + str(pair_m), session, 'Mordor')
        subject_g_tuple = ('pair' + str(pair_g), session, 'Gondor')
        if subject_m_tuple not in MISSING_AUDIO_DATA and subject_g_tuple not in MISSING_AUDIO_DATA:

            # Select corresponding rows of pros_df.
            mask_m = (pros_df.pair == pair_m) & (pros_df.session == session) & (pros_df.lab == 'Mordor')
            mask_g = (pros_df.pair == pair_g) & (pros_df.session == session) & (pros_df.lab == 'Gondor')

            # Sanity checks - there should be exactly one match for Mordor and Gondor rows.
            assert sum(mask_m) == 1, 'Wrong mask for Mordor participant audio in pros_df!'
            assert sum(mask_g) == 1, 'Wrong mask for Gondor participant audio in pros_df!'

            # Transform boolean pandas Series into simple integer for indexing other types of objects as well later.
            idx_m = mask_m[mask_m].index.values[0]
            idx_g = mask_g[mask_g].index.values[0]

            # Get speech timeseries data
            speech_ts_m = pros_df.loc[idx_m, 'speech_timeseries']  # speech_ts is binary
            speech_ts_g = pros_df.loc[idx_g, 'speech_timeseries']
            # Get subtitle objects
            subs_m = pros_df.loc[idx_m, 'srt_segment_info']
            subs_g = pros_df.loc[idx_g, 'srt_segment_info']

            # Prepare subs for calculating vars of interest
            joint_srts = prepare_subs(subs_m, subs_g)
            srt_m, srt_g, srt_idx_m, srt_idx_g = srt_per_speaker(joint_srts)
            srt_idx_m = np.asarray(srt_idx_m)
            srt_idx_g = np.asarray(srt_idx_g)

            # Get session length from srt-s
            session_len_min = joint_srts[-1].end.total_seconds() / 60

            # Get number and rate of turns per speaker.
            turn_no_m = len(srt_m)
            turn_no_g = len(srt_g)
            turn_rate_m = len(srt_m) / session_len_min
            turn_rate_g = len(srt_g) / session_len_min

            # Get turn length stats
            mean_l_m, median_l_m, std_l_m, subs_len_m = get_turn_length_stats(srt_m)[0:4]
            mean_l_g, median_l_g, std_l_g, subs_len_g = get_turn_length_stats(srt_g)[0:4]
            coeff_var_l_m = std_l_m / mean_l_m
            coeff_var_l_g = std_l_g / mean_l_g
            turn_len_autocorr_m = np.corrcoef(subs_len_m[:-1], subs_len_m[1:])[0, 1]
            turn_len_autocorr_g = np.corrcoef(subs_len_g[:-1], subs_len_g[1:])[0, 1]
            turn_len_corr_m, turn_len_corr_g = get_srt_cross_corrs(subs_len_m, subs_len_g, srt_idx_m, srt_idx_g)

            # Get backchannel stats
            backchannels, backchannel_indices = get_backchannels(joint_srts, max_length_s=1.0, padding_s=0.25)
            backch_m, backch_g = srt_per_speaker(backchannels)[0:2]
            backch_no_m = len(backch_m)
            backch_no_g = len(backch_g)
            backch_rate_m = len(backch_m) / session_len_min
            backch_rate_g = len(backch_g) / session_len_min

            # Get joint srt object list without backchannels.
            joint_srts_no_backch = [s for idx, s in enumerate(joint_srts) if idx not in backchannel_indices]

            # Get response times stats
            resp_t_m, resp_idx_m, resp_t_g, resp_idx_g = get_resp_times(joint_srts_no_backch,
                                                                        set_negative_to_zero=False)[2:]
            mean_resp_time_m = np.mean(resp_t_m)
            std_resp_time_m = np.std(resp_t_m)
            coeff_var_resp_time_m = np.std(resp_t_m) / np.mean(resp_t_m)
            mean_resp_time_g = np.mean(resp_t_g)
            std_resp_time_g = np.std(resp_t_g)
            coeff_var_resp_time_g = np.std(resp_t_g) / np.mean(resp_t_g)
            resp_time_autocorr_m = np.corrcoef(resp_t_m[:-1], resp_t_m[1:])[0, 1]
            resp_time_autocorr_g = np.corrcoef(resp_t_g[:-1], resp_t_g[1:])[0, 1]
            resp_time_corr_m, resp_time_corr_g = get_srt_cross_corrs(resp_t_m, resp_t_g, resp_idx_m, resp_idx_g)

            # Get interruptions.
            interrupt_indices, interrupt_idx_m, interrupt_idx_g = get_interrupts(joint_srts)
            interrupt_no_m = len(interrupt_idx_m)
            interrupt_no_g = len(interrupt_idx_g)
            interrupt_rate_m = len(interrupt_idx_m) / session_len_min
            interrupt_rate_g = len(interrupt_idx_g) / session_len_min

            # Get pauses.
            pauses_sum_m, pause_no_m, speech_sum_m = get_pauses_in_speech_turns(speech_ts_m, srt_m,
                                                                                pause_min_len_s=0.2,
                                                                                speech_min_len_s=0.2)
            pauses_sum_g, pause_no_g, speech_sum_g = get_pauses_in_speech_turns(speech_ts_g, srt_g,
                                                                                pause_min_len_s=0.2,
                                                                                speech_min_len_s=0.2)

            # Store vars in dataframe.
            # Speech turn length.
            pros_df.loc[idx_m, 'mean_turn_length'] = mean_l_m
            pros_df.loc[idx_m, 'std_turn_length'] = std_l_m
            pros_df.loc[idx_m, 'coeff_of_var_turn_length'] = coeff_var_l_m
            pros_df.loc[idx_m, 'autocorr_turn_length'] = turn_len_autocorr_m
            pros_df.loc[idx_m, 'crosscorr_turn_length'] = turn_len_corr_m
            pros_df.loc[idx_g, 'mean_turn_length'] = mean_l_g
            pros_df.loc[idx_g, 'std_turn_length'] = std_l_g
            pros_df.loc[idx_g, 'coeff_of_var_turn_length'] = coeff_var_l_g
            pros_df.loc[idx_g, 'autocorr_turn_length'] = turn_len_autocorr_g
            pros_df.loc[idx_g, 'crosscorr_turn_length'] = turn_len_corr_g
            # Response times
            pros_df.loc[idx_m, 'mean_resp_time'] = mean_resp_time_m
            pros_df.loc[idx_m, 'std_resp_time'] = std_resp_time_m
            pros_df.loc[idx_m, 'coeff_of_var_resp_time'] = coeff_var_resp_time_m
            pros_df.loc[idx_m, 'autocorr_resp_time'] = resp_time_autocorr_m
            pros_df.loc[idx_m, 'crosscorr_resp_time'] = resp_time_corr_m
            pros_df.loc[idx_g, 'mean_resp_time'] = mean_resp_time_g
            pros_df.loc[idx_g, 'std_resp_time'] = std_resp_time_g
            pros_df.loc[idx_g, 'coeff_of_var_resp_time'] = coeff_var_resp_time_g
            pros_df.loc[idx_g, 'autocorr_resp_time'] = resp_time_autocorr_g
            pros_df.loc[idx_g, 'crosscorr_resp_time'] = resp_time_corr_g
            # Backchannels
            pros_df.loc[idx_m, 'backch_no'] = backch_no_m
            pros_df.loc[idx_m, 'backch_rate'] = backch_rate_m
            pros_df.loc[idx_g, 'backch_no'] = backch_no_g
            pros_df.loc[idx_g, 'backch_rate'] = backch_rate_g
            # Interruptions
            pros_df.loc[idx_m, 'interrupt_no'] = interrupt_no_m
            pros_df.loc[idx_m, 'interrupt_rate'] = interrupt_rate_m
            pros_df.loc[idx_g, 'interrupt_no'] = interrupt_no_g
            pros_df.loc[idx_g, 'interrupt_rate'] = interrupt_rate_g
            # Number of turns
            pros_df.loc[idx_m, 'turn_no'] = turn_no_m
            pros_df.loc[idx_m, 'turn_rate'] = turn_rate_m
            pros_df.loc[idx_g, 'turn_no'] = turn_no_g
            pros_df.loc[idx_g, 'turn_rate'] = turn_rate_g
            # Pauses
            pros_df.loc[idx_m, 'pause_no'] = np.sum(pause_no_m)
            pros_df.loc[idx_m, 'pause_sum'] = np.sum(pauses_sum_m)
            pros_df.loc[idx_m, 'pause_ratio'] = np.sum(pauses_sum_m) / np.sum(speech_sum_m)
            pros_df.loc[idx_m, 'pause_per_speech_min'] = np.sum(pause_no_m) / (np.sum(speech_sum_m) / 60)
            pros_df.loc[idx_m, 'pause_sum_per_turn'] = np.sum(pause_no_m) / len(srt_m)
            pros_df.loc[idx_g, 'pause_no'] = np.sum(pause_no_g)
            pros_df.loc[idx_g, 'pause_sum'] = np.sum(pauses_sum_g)
            pros_df.loc[idx_g, 'pause_ratio'] = np.sum(pauses_sum_g) / np.sum(speech_sum_g)
            pros_df.loc[idx_g, 'pause_per_speech_min'] = np.sum(pause_no_g) / (np.sum(speech_sum_g) / 60)
            pros_df.loc[idx_g, 'pause_sum_per_turn'] = p.sum(pause_no_g) / len(srt_g)



