"""
Scripts, functions, bits-and-pieces to collect prosody synchrony vars into the big common stats table format.

Variables to include:

(1) Moving window correlation of pitch, intensity, speech_time, speech_syl, speech_rate_syl.

(2) Change over time in intensity, pitch, speech_rate_syl, speech_time_abs, std_int, std_pitch

"""


import pandas as pd
import numpy as np
import os
import pickle
import scipy


DATA_DIR = '/media/adamb/data_disk/prosody_data/'
DF_PKL_WITH_SPEECH_VARS = 'prosody_w_speech_df.pkl'
INTENSITY_TIMESERIES_PKL = 'intensity_timeseries.pkl'
PITCH_TIMESERIES_PKL = 'pitch_timeseries.pkl'
SYLLABLE_TIMESERIES_PKL = 'syllable_timeseries.pkl'
BASE_TABLE_CSV = 'speech_int_pitch_data.csv'
BASE_TABLE_INTERIM_CSV = 'speech_int_pitch_data_interim.csv'
BASE_TABLE_FINAL_CSV = 'speech_int_pitch_synch_data.csv'
# Known cases of missing data
MISSING_AUDIO_DATA = [('pair100', 'BG4', 'Mordor'),
                      ('pair100', 'BG4', 'Gondor'),
                      ('pair188', 'BG5', 'Mordor'),
                      ('pair188', 'BG5', 'Gondor')]
BASE_TABLE_CSV_OUT = 'speech_int_pitch_synch_data.csv'
TIMESERIES_SAMPLING_RATE = 200


def mov_window_corr_conv(ts1, ts2, sr=TIMESERIES_SAMPLING_RATE, window_size_s=60, step_size_s=20,
                         min_ts_size_s=240, stat='mean', only_nonzero=False):
    """
    Helper function calculating correlation and convergence measures on moving window stats (e.g. mean values)
    for two timeseries, ts1 and ts2.

    Measures:
    (1) Correlation across moving window stat vectors.
    (2) Linear regression coefficient across moving window vectors.
    (3) Change in absolute difference between the first and last windows.

    :param ts1:           First timeseries as 1-D numpy array.
    :param ts2:           Second timeseries as 1-D numpy array.
    :param sr:            Sampling rate of both timeseries in Hz. Defaults to module constant TIMESERIES_SAMPLING_RATE.
    :param window_size_s: Window size in seconds. Defaults to 60 s.
    :param step_size_s:   Step size in seconds. Defaults to 20 s.
    :param min_ts_size_s: Minimum size of timeseries to consider. If either "ts1" or "ts2" are shorter than this,
                          the returned values are NaN. Defaults to 240 (4 minutes).
    :param stat:          String, statistic to calculate on windows. One of ['mean', 'median', 'std', 'sum'].
                          Defaults to "mean".
    :param only_nonzero:  Boolean. If True, only nonzero values are taken into account when calculating the selected
                          statistic for each window. Defaults to False.
    :return: corr_coeff:         Pearson correlation coefficient across the two window statistic arrays.
    :return: conv_beta:          Linear coefficient for the absolute differences across window statistics over time.
                                 That is, it captures if the window statistics become more similar across the timeseries
                                 over time or not.
    :return: conv_first_to_last: Change in the difference between the last and first window statistics.
    :return: ts1_stats:          Statistic per window for timeseries 1.
    :return: ts2_stats:          Statistic per window for timeseries 2.

    """
    # Define window and step sizes in terms of samples.
    window_size_samples = round(sr * window_size_s)
    step_size_samples = round(sr * step_size_s)

    # If ts1 and ts2 have different length, trim to shorter.
    if len(ts1) != len(ts2):
        min_l = min(len(ts1), len(ts2))
        if len(ts1) > len(ts2):
            ts1 = ts1[:min_l]
        elif len(ts1) < len(ts2):
            ts2 = ts2[:min_l]

    # Check size - do we have enough data at all?
    if len(ts1) < round(min_ts_size_s * sr):
        # print('Time series too short!')
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Derive moving window start and endpoints in terms of samples.
    start_samples = np.arange(0, len(ts1)-window_size_samples+0.1, step_size_samples).astype(int)
    end_samples = np.arange(window_size_samples, len(ts1)+0.1, step_size_samples).astype(int)

    # Loop through windows, calculate stat, and
    ts1_l = []
    ts2_l = []
    for s in zip(start_samples, end_samples):

        # Samples for current window
        ts1_sample = ts1[s[0]:s[1]]
        ts2_sample = ts2[s[0]:s[1]]
        # Check for nonzero?
        if only_nonzero:
            ts1_sample = ts1_sample[np.nonzero(ts1_sample)]
            ts2_sample = ts2_sample[np.nonzero(ts2_sample)]
    #            if len(ts1_sample) == 0:
    #                ts1_sample = np.nan
    #            if not len(ts2_sample) == 0:
    #                ts2_sample = np.nan

        # Calculate stat for window
        if stat == 'mean':
            ts1_l.append(np.mean(ts1_sample))
            ts2_l.append(np.mean(ts2_sample))
        elif stat == 'median':
            ts1_l.append(np.median(ts1_sample))
            ts2_l.append(np.median(ts2_sample))
        elif stat == 'std':
            ts1_l.append(np.std(ts1_sample))
            ts2_l.append(np.std(ts2_sample))
        elif stat == 'sum':
            ts1_l.append(np.sum(ts1_sample))
            ts2_l.append(np.sum(ts2_sample))

    # Turn lists into np arrays before estimating correlation / convergence.
    ts1_stats = np.asarray(ts1_l)
    ts2_stats = np.asarray(ts2_l)

    # Filter potential nan values. Get masks for valid elements.
    mask1 = ~np.isnan(ts1_stats)
    mask2 = ~np.isnan(ts2_stats)
    mask_both = mask1 & mask2
    ts1_stats = ts1_stats[mask_both]
    ts2_stats = ts2_stats[mask_both]

    # Simple correlation
    corr_coeff = np.corrcoef(ts1_stats, ts2_stats)[0, 1]
    # Linear coefficient of the differences over time.
    stats_diff = np.abs(ts1_stats-ts2_stats)
    x = np.arange(0, len(stats_diff), 1)
    conv_beta = (np.dot(x - np.mean(x), stats_diff - np.mean(stats_diff))) / np.dot(x - np.mean(x), x - np.mean(x))  # Simple linear regression coeff
    # Change in absolute difference between first and last windows.
    conv_first_to_last = stats_diff[-1] - stats_diff[0]

    return corr_coeff, conv_beta, conv_first_to_last, ts1_stats, ts2_stats


def real_vs_pseudo(stat_df, target_var, real_mask, pseudo_mask):
    print('\nTesting ' + target_var + '.')
    real_values = stat_df.loc[real_mask, target_var].to_numpy()
    pseudo_values = stat_df.loc[pseudo_mask, target_var].to_numpy()
    res = scipy.stats.ttest_ind(real_values, pseudo_values, equal_var=False, nan_policy='omit')
    print('Sample means (SD), for real and pseudo: ')
    print(str(round(np.nanmean(real_values), 3)) + '(' + str(round(np.nanstd(real_values), 3)) + '); ' +
          str(round(np.nanmean(pseudo_values), 3)) + '(' + str(round(np.nanstd(pseudo_values), 3)) + ')')
    print(res)

    return res


# Load csv
stat_df = pd.read_csv(os.path.join(DATA_DIR, BASE_TABLE_CSV))
# Add new vars / columns
stat_df.loc[:, 'corr_speech_time_abs'] = np.nan
stat_df.loc[:, 'conv_b_speech_time_abs'] = np.nan
stat_df.loc[:, 'diff_change_speech_time_abs'] = np.nan
stat_df.loc[:, 'corr_speech_rate_syl'] = np.nan
stat_df.loc[:, 'conv_b_speech_rate_syl'] = np.nan
stat_df.loc[:, 'diff_change_speech_rate_syl'] = np.nan
stat_df.loc[:, 'corr_speech_syl'] = np.nan
stat_df.loc[:, 'conv_b_speech_syl'] = np.nan
stat_df.loc[:, 'diff_change_speech_syl'] = np.nan
stat_df.loc[:, 'corr_pitch'] = np.nan
stat_df.loc[:, 'conv_b_pitch'] = np.nan
stat_df.loc[:, 'diff_change_pitch'] = np.nan
stat_df.loc[:, 'corr_std_pitch'] = np.nan
stat_df.loc[:, 'conv_b_std_pitch'] = np.nan
stat_df.loc[:, 'diff_change_std_pitch'] = np.nan
stat_df.loc[:, 'corr_int'] = np.nan
stat_df.loc[:, 'conv_b_int'] = np.nan
stat_df.loc[:, 'diff_change_int'] = np.nan
stat_df.loc[:, 'corr_std_int'] = np.nan
stat_df.loc[:, 'conv_b_std_int'] = np.nan
stat_df.loc[:, 'diff_change_std_int'] = np.nan

# Load intensity data with pickle load.
with open(os.path.join(DATA_DIR, INTENSITY_TIMESERIES_PKL), 'rb') as f:
    int_data = pickle.load(f)
int_indices = int_data[0]
int_ts = int_data[1]

# Load pitch data with pickle load.
with open(os.path.join(DATA_DIR, PITCH_TIMESERIES_PKL), 'rb') as f:
    pitch_data = pickle.load(f)
pitch_indices = pitch_data[0]
pitch_ts = pitch_data[1]

# Load syllable data with pickle load.
with open(os.path.join(DATA_DIR, SYLLABLE_TIMESERIES_PKL), 'rb') as f:
    syl_data = pickle.load(f)
syl_indices = syl_data[0]
syl_ts = syl_data[1]

# Load df with prosody vars
pros_df = pd.read_pickle(os.path.join(DATA_DIR, DF_PKL_WITH_SPEECH_VARS))


###################################################################################
##############    SPEECH AMOUNT AND SPEECH RATE SYNCH VARIABLES ###################
###################################################################################


# Temporary parameters for moving windows.
w_s = 25
s_s = 12.5
min_s = 237.5

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

        ################################################################
        # Synch variables for time spent speaking.

        # Select speech timeseries for both participants first.
        speech_ts_m = pros_df.loc[idx_m, 'speech_timeseries']
        speech_ts_g = pros_df.loc[idx_g, 'speech_timeseries']

        # Correlation and convergence in terms of the amount of speech (temporally).
        corr_coeff, conv_beta, diff_change,\
            speech_stats1, speech_stats2 = mov_window_corr_conv(speech_ts_m, speech_ts_g,
                                                                sr=TIMESERIES_SAMPLING_RATE,
                                                                window_size_s=w_s,
                                                                step_size_s=s_s,
                                                                min_ts_size_s=min_s,
                                                                stat='sum',
                                                                only_nonzero=False)

        stat_df.loc[pair_idx, 'corr_speech_time_abs'] = corr_coeff
        stat_df.loc[pair_idx, 'conv_b_speech_time_abs'] = conv_beta
        stat_df.loc[pair_idx, 'diff_change_speech_time_abs'] = diff_change

        ################################################################
        # Synch variables for amount of speech (syllables spoken).

        # First sanity check - are the pros_df indices matching the syllable ones? They should, just to be sure.
        assert idx_m == syl_indices[idx_m], 'Mismatch between pros_df and syllable data indices for Mordor participant!'
        assert idx_g == syl_indices[idx_g], 'Mismatch between pros_df and syllable data indices for Gondor participant!'

        # Select speech rate / syllable timeseries for both participants.
        syl_ts_m = syl_ts[idx_m]
        syl_ts_g = syl_ts[idx_g]

        # Correlation / convergence in terms of the amount of speech (number of syllables).
        corr_coeff, conv_beta, diff_change,\
            syl_stats1, syl_stats2 = mov_window_corr_conv(syl_ts_m, syl_ts_g,
                                                          sr=TIMESERIES_SAMPLING_RATE,
                                                          window_size_s=w_s,
                                                          step_size_s=s_s,
                                                          min_ts_size_s=min_s,
                                                          stat='sum',
                                                          only_nonzero=False)

        stat_df.loc[pair_idx, 'corr_speech_syl'] = corr_coeff
        stat_df.loc[pair_idx, 'conv_b_speech_syl'] = conv_beta
        stat_df.loc[pair_idx, 'diff_change_speech_syl'] = diff_change

        ################################################################
        # Synch variables for speech rate (syl/s).

        if isinstance(syl_stats1, np.ndarray) and isinstance(syl_stats2, np.ndarray):
            # Calculate speech rate / syllable rate per moving window, from existing speech and syllable window values.
            speech_stats1 = speech_stats1.astype(float)
            speech_stats2 = speech_stats2.astype(float)
            speech_stats1[speech_stats1 == 0] = np.inf
            speech_stats2[speech_stats2 == 0] = np.inf
            speech_rate_stats1 = np.divide(syl_stats1, speech_stats1)
            speech_rate_stats2 = np.divide(syl_stats2, speech_stats2)

            speech_stats1[speech_stats1 == np.inf] = 0
            speech_stats2[speech_stats2 == np.inf] = 0

            # Linear coefficient of the differences over time.
            stats_diff = np.abs(speech_rate_stats1 - speech_rate_stats2)
            x = np.arange(0, len(stats_diff), 1)
            conv_beta = (np.dot(x - np.mean(x), stats_diff - np.mean(stats_diff))) / \
                np.dot(x - np.mean(x), x - np.mean(x))  # Simple linear regression coeff
            # Change in absolute difference between first and last windows.
            conv_first_to_last = stats_diff[-1] - stats_diff[0]

            stat_df.loc[pair_idx, 'corr_speech_rate_syl'] = np.corrcoef(speech_rate_stats1, speech_rate_stats2)[0, 1]
            stat_df.loc[pair_idx, 'conv_b_speech_rate_syl'] = conv_beta
            stat_df.loc[pair_idx, 'diff_change_speech_rate_syl'] = conv_first_to_last
        else:
            stat_df.loc[pair_idx, 'corr_speech_rate_syl'] = np.nan
            stat_df.loc[pair_idx, 'conv_b_speech_rate_syl'] = np.nan
            stat_df.loc[pair_idx, 'diff_change_speech_rate_syl'] = np.nan


real_mask = stat_df.type_label == 'real'
pseudo_mask = stat_df.type_label == 'pseudo'

res_s1 = real_vs_pseudo(stat_df, 'corr_speech_time_abs', real_mask, pseudo_mask)
res_s2 = real_vs_pseudo(stat_df, 'conv_b_speech_time_abs', real_mask, pseudo_mask)
res_s3 = real_vs_pseudo(stat_df, 'diff_change_speech_time_abs', real_mask, pseudo_mask)

res_s4 = real_vs_pseudo(stat_df, 'corr_speech_syl', real_mask, pseudo_mask)
res_s5 = real_vs_pseudo(stat_df, 'conv_b_speech_syl', real_mask, pseudo_mask)
res_s6 = real_vs_pseudo(stat_df, 'diff_change_speech_syl', real_mask, pseudo_mask)

res_s7 = real_vs_pseudo(stat_df, 'corr_speech_rate_syl', real_mask, pseudo_mask)
res_s8 = real_vs_pseudo(stat_df, 'conv_b_speech_rate_syl', real_mask, pseudo_mask)
res_s9 = real_vs_pseudo(stat_df, 'diff_change_speech_rate_syl', real_mask, pseudo_mask)

# Interim save
with open(os.path.join(DATA_DIR, BASE_TABLE_INTERIM_CSV), 'w') as f:
    stat_df.to_csv(f)


###########################################################
##############    PITCH SYNCH VARIABLES ###################
###########################################################


# Temporary parameters for moving windows.
w_s = 25
s_s = 12.5
min_s = 237.5

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

        ################################################################
        # Synch variables for mean and std pitch.

        # First sanity check - are the pros_df indices matching the pitch ones? They should, just to be sure.
        assert idx_m == pitch_indices[idx_m], 'Mismatch between pros_df and pitch data indices for Mordor participant!'
        assert idx_g == pitch_indices[idx_g], 'Mismatch between pros_df and pitch data indices for Gondor participant!'

        # Select speech rate / syllable timeseries for both participants.
        pitch_ts_m = pitch_ts[idx_m]
        pitch_ts_g = pitch_ts[idx_g]

        # Correlation / convergence in terms of pitch.
        # Using mean as stat.
        corr_coeff, conv_beta, diff_change = mov_window_corr_conv(pitch_ts_m, pitch_ts_g,
                                                                  sr=TIMESERIES_SAMPLING_RATE,
                                                                  window_size_s=w_s,
                                                                  step_size_s=s_s,
                                                                  min_ts_size_s=min_s,
                                                                  stat='mean',
                                                                  only_nonzero=True)[0:3]

        stat_df.loc[pair_idx, 'corr_pitch'] = corr_coeff
        stat_df.loc[pair_idx, 'conv_b_pitch'] = conv_beta
        stat_df.loc[pair_idx, 'diff_change_pitch'] = diff_change

        # Correlation / convergence in terms of pitch.
        # Using std as stat.
        corr_coeff, conv_beta, diff_change = mov_window_corr_conv(pitch_ts_m, pitch_ts_g,
                                                                  sr=TIMESERIES_SAMPLING_RATE,
                                                                  window_size_s=w_s,
                                                                  step_size_s=s_s,
                                                                  min_ts_size_s=min_s,
                                                                  stat='std',
                                                                  only_nonzero=True)[0:3]

        stat_df.loc[pair_idx, 'corr_std_pitch'] = corr_coeff
        stat_df.loc[pair_idx, 'conv_b_std_pitch'] = conv_beta
        stat_df.loc[pair_idx, 'diff_change_std_pitch'] = diff_change


real_mask = stat_df.type_label == 'real'
pseudo_mask = stat_df.type_label == 'pseudo'

res_p1 = real_vs_pseudo(stat_df, 'corr_pitch', real_mask, pseudo_mask)
res_p2 = real_vs_pseudo(stat_df, 'conv_b_pitch', real_mask, pseudo_mask)
res_p3 = real_vs_pseudo(stat_df, 'diff_change_pitch', real_mask, pseudo_mask)

res_p4 = real_vs_pseudo(stat_df, 'corr_std_pitch', real_mask, pseudo_mask)
res_p5 = real_vs_pseudo(stat_df, 'conv_b_std_pitch', real_mask, pseudo_mask)
res_p6 = real_vs_pseudo(stat_df, 'diff_change_std_pitch', real_mask, pseudo_mask)


###############################################################
##############    INTENSITY SYNCH VARIABLES ###################
###############################################################


# Temporary parameters for moving windows.
w_s = 25
s_s = 12.5
min_s = 237.5

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

        ################################################################
        # Synch variables for mean and std intensity.

        # First sanity check - are the pros_df indices matching the intensity ones? They should, just to be sure.
        assert idx_m == int_indices[idx_m], 'Mismatch between pros_df and intensity data indices for Mordor participant!'
        assert idx_g == int_indices[idx_g], 'Mismatch between pros_df and intensity data indices for Gondor participant!'

        # Select speech rate / syllable timeseries for both participants.
        int_ts_m = int_ts[idx_m]
        int_ts_g = int_ts[idx_g]

        # Correlation / convergence in terms of vocal intensity.
        # Using mean as stat.
        corr_coeff, conv_beta, diff_change = mov_window_corr_conv(int_ts_m, int_ts_g,
                                                                  sr=TIMESERIES_SAMPLING_RATE,
                                                                  window_size_s=w_s,
                                                                  step_size_s=s_s,
                                                                  min_ts_size_s=min_s,
                                                                  stat='mean',
                                                                  only_nonzero=True)[0:3]

        stat_df.loc[pair_idx, 'corr_int'] = corr_coeff
        stat_df.loc[pair_idx, 'conv_b_int'] = conv_beta
        stat_df.loc[pair_idx, 'diff_change_int'] = diff_change

        # Correlation / convergence in terms of  vocal intensity.
        # Using std as stat.
        corr_coeff, conv_beta, diff_change = mov_window_corr_conv(int_ts_m, int_ts_g,
                                                                  sr=TIMESERIES_SAMPLING_RATE,
                                                                  window_size_s=w_s,
                                                                  step_size_s=s_s,
                                                                  min_ts_size_s=min_s,
                                                                  stat='std',
                                                                  only_nonzero=True)[0:3]

        stat_df.loc[pair_idx, 'corr_std_int'] = corr_coeff
        stat_df.loc[pair_idx, 'conv_b_std_int'] = conv_beta
        stat_df.loc[pair_idx, 'diff_change_std_int'] = diff_change


real_mask = stat_df.type_label == 'real'
pseudo_mask = stat_df.type_label == 'pseudo'

res_i1 = real_vs_pseudo(stat_df, 'corr_int', real_mask, pseudo_mask)
res_i2 = real_vs_pseudo(stat_df, 'conv_b_int', real_mask, pseudo_mask)
res_i3 = real_vs_pseudo(stat_df, 'diff_change_int', real_mask, pseudo_mask)

res_i4 = real_vs_pseudo(stat_df, 'corr_std_int', real_mask, pseudo_mask)
res_i5 = real_vs_pseudo(stat_df, 'conv_b_std_int', real_mask, pseudo_mask)
res_i6 = real_vs_pseudo(stat_df, 'diff_change_std_int', real_mask, pseudo_mask)

# Final save
with open(os.path.join(DATA_DIR, BASE_TABLE_FINAL_CSV), 'w') as f:
    stat_df.to_csv(f)
