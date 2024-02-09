"""
Scripts, functions, bits-and-pieces to collect prosody data into the big common stats table format.

Variables to include:

(1) Mean and diff of speech_time_abs, speech_time_rel, speech_syl, speech_rate_syl, pitch, and intensity.

(2) Moving window correlation of pitch, intensity, speech_time, speech_syl, speech_rate_syl.

(3) Change over time in intensity, pitch, speech_rate_syl

"""


import pandas as pd
import numpy as np
import os
import pickle


DATA_DIR = '/media/adamb/data_disk/prosody_data/'
DF_PKL_WITH_SPEECH_VARS = 'prosody_w_speech_df.pkl'
INTENSITY_TIMESERIES_PKL = 'intensity_timeseries.pkl'
PITCH_TIMESERIES_PKL = 'pitch_timeseries.pkl'
BASE_TABLE_CSV = 'base_data_form.csv'
# Known cases of missing data
MISSING_AUDIO_DATA = [('pair100', 'BG4', 'Mordor'),
                      ('pair100', 'BG4', 'Gondor'),
                      ('pair188', 'BG5', 'Mordor'),
                      ('pair188', 'BG5', 'Gondor')]
BASE_TABLE_CSV_OUT = 'speech_int_pitch_data.csv'
DF_PKL_WITH_SPEECH_INT_PITCH_VARS = 'prosody_w_speech_int_pitch_df.pkl'


# Load csv
stat_df = pd.read_csv(os.path.join(DATA_DIR, BASE_TABLE_CSV))
# Add new vars / columns
stat_df.loc[:, 'mean_speech_time_abs'] = None
stat_df.loc[:, 'mean_speech_time_rel'] = None
stat_df.loc[:, 'mean_speech_syl'] = None
stat_df.loc[:, 'mean_speech_rate_syl'] = None
stat_df.loc[:, 'mean_pitch'] = None
stat_df.loc[:, 'mean_int'] = None
stat_df.loc[:, 'mean_std_pitch'] = None
stat_df.loc[:, 'mean_std_int'] = None
stat_df.loc[:, 'diff_speech_time_abs'] = None
stat_df.loc[:, 'diff_speech_time_rel'] = None
stat_df.loc[:, 'diff_speech_syl'] = None
stat_df.loc[:, 'diff_speech_rate_syl'] = None
stat_df.loc[:, 'diff_pitch'] = None
stat_df.loc[:, 'diff_int'] = None
stat_df.loc[:, 'diff_std_pitch'] = None
stat_df.loc[:, 'diff_std_int'] = None

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

# Load df with prosody vars
pros_df = pd.read_pickle(os.path.join(DATA_DIR, DF_PKL_WITH_SPEECH_VARS))
pros_df.loc[:, 'mean_int'] = None
pros_df.loc[:, 'std_int'] = None
pros_df.loc[:, 'mean_pitch'] = None
pros_df.loc[:, 'std_pitch'] = None

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

        # Simple mean and diff variables from speech vars
        stat_df.loc[pair_idx, 'mean_speech_time_abs'] = (pros_df.loc[idx_m, 'speech_time_abs'] +
                                                         pros_df.loc[idx_g, 'speech_time_abs']) / 2
        stat_df.loc[pair_idx, 'mean_speech_time_rel'] = (pros_df.loc[idx_m, 'speech_time_rel'] +
                                                         pros_df.loc[idx_g, 'speech_time_rel']) / 2
        stat_df.loc[pair_idx, 'mean_speech_syl'] = (pros_df.loc[idx_m, 'speech_syl'] +
                                                    pros_df.loc[idx_g, 'speech_syl']) / 2
        stat_df.loc[pair_idx, 'mean_speech_rate_syl'] = (pros_df.loc[idx_m, 'speech_rate_syl'] +
                                                         pros_df.loc[idx_g, 'speech_rate_syl']) / 2
        stat_df.loc[pair_idx, 'diff_speech_time_abs'] = abs(pros_df.loc[idx_m, 'speech_time_abs'] -
                                                            pros_df.loc[idx_g, 'speech_time_abs'])
        stat_df.loc[pair_idx, 'diff_speech_time_rel'] = abs(pros_df.loc[idx_m, 'speech_time_rel'] -
                                                            pros_df.loc[idx_g, 'speech_time_rel'])
        stat_df.loc[pair_idx, 'diff_speech_syl'] = abs(pros_df.loc[idx_m, 'speech_syl'] -
                                                       pros_df.loc[idx_g, 'speech_syl'])
        stat_df.loc[pair_idx, 'diff_speech_rate_syl'] = abs(pros_df.loc[idx_m, 'speech_rate_syl'] -
                                                            pros_df.loc[idx_g, 'speech_rate_syl'])

        ##########################################
        # Simple mean and diff variables for intensity.

        # First sanity check - are the pros_df indices matching the intensity ones? They should, just to be sure.
        assert idx_m == int_indices[idx_m], 'Mismatch between pros_df and intensity data indices for Mordor participant!'
        assert idx_g == int_indices[idx_g], 'Mismatch between pros_df and intensity data indices for Gondor participant!'

        # Get mean and std intensity for Mordor participant, store in prosody df.
        speech_ts_m = pros_df.loc[idx_m, 'speech_timeseries']
        int_ts_m = int_ts[idx_m]
        assert len(speech_ts_m) == len(int_ts_m), 'Speech and intensity timeseries have different length for Mordor part.!'
        int_speech_m = int_ts_m[speech_ts_m.astype(bool)]
        pros_df.loc[idx_m, 'mean_int'] = np.mean(int_speech_m)
        pros_df.loc[idx_m, 'std_int'] = np.std(int_speech_m)

        # Get mean and std intensity for Gondor participant, store in prosody df.
        speech_ts_g = pros_df.loc[idx_g, 'speech_timeseries']
        int_ts_g = int_ts[idx_g]
        assert len(speech_ts_g) == len(int_ts_g), 'Speech and intensity timeseries have different length for Mordor part.!'
        int_speech_g = int_ts_g[speech_ts_g.astype(bool)]
        pros_df.loc[idx_g, 'mean_int'] = np.mean(int_speech_g)
        pros_df.loc[idx_g, 'std_int'] = np.std(int_speech_g)

        # Finally, get mean and diff intensity values for pair.
        stat_df.loc[pair_idx, 'mean_int'] = (pros_df.loc[idx_m, 'mean_int'] +
                                             pros_df.loc[idx_g, 'mean_int']) / 2
        stat_df.loc[pair_idx, 'mean_std_int'] = (pros_df.loc[idx_m, 'std_int'] +
                                                 pros_df.loc[idx_g, 'std_int']) / 2
        stat_df.loc[pair_idx, 'diff_int'] = abs(pros_df.loc[idx_m, 'mean_int'] -
                                                pros_df.loc[idx_g, 'mean_int'])
        stat_df.loc[pair_idx, 'diff_std_int'] = abs(pros_df.loc[idx_m, 'std_int'] -
                                                    pros_df.loc[idx_g, 'std_int'])

        ##########################################
        # Simple mean and diff variables for pitch.

        # First sanity check - are the pros_df indices matching the pitch ones? They should, just to be sure.
        assert idx_m == pitch_indices[idx_m], 'Mismatch between pros_df and pitch data indices for Mordor participant!'
        assert idx_g == pitch_indices[idx_g], 'Mismatch between pros_df and pitch data indices for Gondor participant!'

        # Get mean and std pitch for Mordor participant, store in prosody df.
        speech_ts_m = pros_df.loc[idx_m, 'speech_timeseries']
        pitch_ts_m = pitch_ts[idx_m]
        assert len(speech_ts_m) == len(pitch_ts_m), 'Speech and pitch timeseries have different length for Mordor part.!'
        pitch_speech_m = pitch_ts_m[speech_ts_m.astype(bool)]
        pros_df.loc[idx_m, 'mean_pitch'] = np.mean(pitch_speech_m)
        pros_df.loc[idx_m, 'std_pitch'] = np.std(pitch_speech_m)

        # Get mean and std pitch for Gondor participant, store in prosody df.
        speech_ts_g = pros_df.loc[idx_g, 'speech_timeseries']
        pitch_ts_g = pitch_ts[idx_g]
        assert len(speech_ts_g) == len(pitch_ts_g), 'Speech and pitch timeseries have different length for Mordor part.!'
        pitch_speech_g = pitch_ts_g[speech_ts_g.astype(bool)]
        pros_df.loc[idx_g, 'mean_pitch'] = np.mean(pitch_speech_g)
        pros_df.loc[idx_g, 'std_pitch'] = np.std(pitch_speech_g)

        # Finally, get mean and diff pitch values for pair.
        stat_df.loc[pair_idx, 'mean_pitch'] = (pros_df.loc[idx_m, 'mean_pitch'] +
                                               pros_df.loc[idx_g, 'mean_pitch']) / 2
        stat_df.loc[pair_idx, 'mean_std_pitch'] = (pros_df.loc[idx_m, 'std_pitch'] +
                                                   pros_df.loc[idx_g, 'std_pitch']) / 2
        stat_df.loc[pair_idx, 'diff_pitch'] = abs(pros_df.loc[idx_m, 'mean_pitch'] -
                                                  pros_df.loc[idx_g, 'mean_pitch'])
        stat_df.loc[pair_idx, 'diff_std_pitch'] = abs(pros_df.loc[idx_m, 'std_pitch'] -
                                                      pros_df.loc[idx_g, 'std_pitch'])

# Write out both the prosody df and the stat_df, to pkl and csv, respectively.
pros_df.to_pickle(os.path.join(DATA_DIR, DF_PKL_WITH_SPEECH_INT_PITCH_VARS))
with open(os.path.join(DATA_DIR, BASE_TABLE_CSV_OUT), 'w') as f:
    stat_df.to_csv(f)









