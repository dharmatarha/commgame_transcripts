"""
RMS-based filtering on audio signals. Hardcoded params in main()

USAGE:
python speech_rms_filtering PAIRNUMBER LABNAME --audio_dir AUDIO_DIR

Expects audio files with the naming convention
"pairPAIRNUMBER_LABNAME_SESSION_repaired_mono_noisered.wav"
(e.g. pair99_Mordor_freeConv_repaired_mono_noisered.wav) in the folder AUDIO_DIR.

Output are wav files saved into AUDIO_DIR, with naming convention:
"pairPAIRNUMBER_LABNAME_SESSION_repaired_mono_noisered_filtered.wav"
(e.g. pair99_Mordor_freeConv_repaired_mono_noisered_filtered.wav)


"""

import os
import argparse
import numpy as np
import librosa as lr
import soundfile as sf
from scipy.signal import medfilt
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def find_speech_files(pair, lab, audio_dir):
    """
    Function to find relevant audio files for RMS-based filtering. Looks up the wav files that
    underwent audio preprocessing and noise reduction (have the ending "*_repaired_mono_noisered.wav)
    for a given participant (defined by pair number and lab name together, e.g. pair99_Mordor), for all sessions
    (that is, for all BG games and for freeConv).

    :param pair:        Int, pair number.
    :param lab:         Str, lab name, one of ['Mordor', 'Gondor']
    :param audio_dir:   Str, path to directory holding all audio files.
    :return session_files: List, where each element is a path to a wav file.
    """
    # Get BG files for pair and lab
    bg = 0
    session_files = []
    final_bg = None
    while not final_bg:
        bg = bg + 1
        session_pattern = '_'.join(['pair'+str(pair), lab, 'BG'+str(bg), 'repaired_mono_noisered.wav'])
        session_path = os.path.join(audio_dir, session_pattern)
        if os.path.exists(session_path):
            session_files.append(session_path)
        else:
            final_bg = bg - 1
    # Get freeConv file
    session_pattern = '_'.join(['pair' + str(pair), lab, 'freeConv', 'repaired_mono_noisered.wav'])
    session_path = os.path.join(audio_dir, session_pattern)
    if os.path.exists(session_path):
        session_files.append(session_path)
    else:
        raise FileNotFoundError('Could not find freeConv file ' + session_pattern)

    return session_files


def get_log_rms(session_files, ref_sampling_rate=44100, n_fft=2048, hop_length=1024):
    """
    Function to calculate Root Mean Square (RMS) of the audio signals in the wav files supplied in session_files.
    RMS is estimated from the STFT (using librosa). Both the RMS values and the indices of frames are returned in
    numpy arrays, collected into lists.

    :param session_files:       List, each element is a path to a wav file (usually output of find_speech_files).
    :param ref_sampling_rate:   Int, expected (reference) sampling rate for wav files.
    :param n_fft:               Int, passed on to the "n_fft" input argument of librosa.stft.
    :param hop_length:          Int, passed on to the "hop_length" input argument of librosa.stft.
    :return log_rms_list:       List, where each element is a numpy array of log10 RMS values (RMS value per frame).
    :return samples_list:       List, where each element is 2D numpy array with shape (2, NUMBER_OF_FRAMES).
                                Each column holds the start and end indices for the corresponding frame.
    """
    # Output var
    log_rms_list = []
    samples_list = []
    for session in session_files:
        # Load wav
        signal, sr = lr.load(session, sr=None)
        # Sanity check
        if sr != ref_sampling_rate:
            raise ValueError('Unexpected sampling rate from file ' + session)
        # Get log RMS from power spectrum
        spectra = lr.stft(y=signal, n_fft=n_fft, hop_length=hop_length, center=False)
        rms_tmp = lr.feature.rms(S=np.abs(spectra) ** 2, frame_length=n_fft)
        rms = np.log10(rms_tmp + np.finfo(np.float32).eps)  # avoid calling log on zero
        # Get start and end samples for each window (frame)
        samples = lr.frames_to_samples(np.arange(0, spectra.shape[1]), hop_length=hop_length, n_fft=n_fft)
        sample_starts = samples - np.floor(n_fft/2)
        sample_ends = samples + np.floor(n_fft/2)
        sample_bounds = np.vstack((sample_starts, sample_ends))

        log_rms_list.append(rms)
        samples_list.append(sample_bounds)

    return log_rms_list, samples_list


def plot_rms_hist(rms):
    """
    Function to draw a simple histogram of log10 RMS values from a list of log10 RMS value arrays.
    Arrays are first stacked together, then matplotlib pyplot hist is called.

    :param rms:  List, where each element is a numpy array of log10 RMS values (usually the output from get_log_rms).
    :return fig: Matplotlib figure handle.
    """
    # Stack all numpy arrays into a vector
    data = np.array([])
    for arr in rms:
        data = np.hstack((data, arr.squeeze(0)))
    fig = plt.figure()
    plt.hist(data, 100)
    plt.show(block=False)
    plt.pause(3)

    return fig


def get_thresh_input(figure):
    """
    Get input for log10 RMS threshold values after drawing the histogram with plot.rms_hist.
    :param figure:          Matplotlib figure handle.
    :return threshold_high: Numeric value, log10 RMS value, answer to the prompt for higher threshold.
    :return threshold_low:  Numeric value, log10 RMS value, answer to the prompt for lower threshold.
    """
    threshold_high = input('\nWhat should be the higher threshold for noise? (in log10 values)\n')
    threshold_high = float(threshold_high)
    plt.pause(1)
    threshold_low = input('\nWhat should be the lower threshold for noise? (in log10 values)\n')
    threshold_low = float(threshold_low)
    plt.close(figure)

    return threshold_high, threshold_low


def clean_signal(session_path, session_rms_log, session_sample_bounds, rms_threshold_log,
                 min_length_s=0.25, ref_sampling_rate=44100, dampening_factor=0.0001):
    """
    !!!  NOT USED BUT LEFT HERE IN CASE WE REVERT TO IT  !!!

    :param session_path:
    :param session_rms_log:
    :param session_sample_bounds:
    :param rms_threshold_log:
    :param min_length_s:
    :param ref_sampling_rate:
    :param dampening_factor:
    :return:
    """
    # Constants
    min_length_samples = np.floor(min_length_s * ref_sampling_rate)
    min_length_frames = np.ceil(min_length_samples / int(np.diff(session_sample_bounds[:, 0])))

    # Init np random generator
    rng = np.random.default_rng()

    # Load wav
    signal, sr = lr.load(session_path, sr=None)
    # Sanity check
    if sr != ref_sampling_rate:
        raise ValueError('Unexpected sampling rate from file ' + session_path)

    # Check for frames below rms_threshold
    low_rms = session_rms_log < rms_threshold_log
    low_rms_ranges = true_runs(low_rms.squeeze(0))

    # Get low rms segment starts and ends in samples
    low_rms_idx = (low_rms_ranges[:, 1] - low_rms_ranges[:, 0]) >= min_length_frames
    ranges_to_clean = low_rms_ranges[low_rms_idx, :]
    samples_to_clean = np.vstack((session_sample_bounds[0, ranges_to_clean[:, 0]],
                                  session_sample_bounds[1, ranges_to_clean[:, 1]-1])).T.astype(np.int64)

    # Go through sample ranges to clean, apply cleaner function
    for idx in range(samples_to_clean.shape[0]):
        segment_start = samples_to_clean[idx, 0]
        segment_end = min(samples_to_clean[idx, 1], signal.size-1)
        tmp = signal[segment_start: segment_end].copy() * dampening_factor
        shuffled_signal = rng.permutation(tmp)
        signal[segment_start: segment_end] = shuffled_signal

    # save out cleaned wav
    new_path = session_path[0:-4] + '_shuffled.wav'
    sf.write(new_path, signal, int(sr), subtype='PCM_16')

    return new_path, samples_to_clean


def true_runs(arr):
    """
    Cool utility that finds runs of True values in the input signal.
    :param arr: Numpy array, boolean, 1-dimensional.
    :return:    Numpy array, with shape (true_runs, 2), with each row containing the range of a
                True run (start and end).
    """
    # Create an array that is 1 where arr is 0, and pad each end with an extra 0.
    is_true = np.concatenate(([False], arr, [False]))
    abs_diff = np.abs(np.diff(is_true))
    # Runs start and end where abs_diff is 1.
    ranges = np.where(abs_diff)[0].reshape(-1, 2)
    return ranges


def rms_weighting_filter(session_path, session_rms_log, session_sample_bounds, rms_threshold_high,
                         rms_threshold_low, ref_sampling_rate=44100, max_log_weight=0, min_log_weight=-4,
                         win_length=11, noise_sigma=0.001, medfilt_size=999):
    """
    RMS-based filtering function. Calls functions windowed_weigthing and weighting_fun.
    It maps the mean RMS values of the windowed signal which fall between the speficied thresholds / cutoffs to 
    the weights between the specified maximum and minimum values proportionally, then multiplies the signal with
    the weight vector. RMS values above / below the maximum / minimum thresholds are assigned the maximum / minimum weights.
    Gaussian noise is added to mask low-RMS signal segments.
    
    
    :param session_path:	   Path to session file (audio)
    :param session_rms_log:	   Numpy array of log10 RMS values, usually from get_log_rms output,
    				   corresponding to session file.	
    :param session_sample_bounds: 2D numpy array with shape (2, NUMBER_OF_FRAMES). Each column holds
    				   the start and end indices for the corresponding frame. Usually from
    				   get_log_rms output, corresponding to session_rms_log.
    :param rms_threshold_high:    Numeric value, higher RMS cutoff (in log10 scale).
    :param rms_threshold_low:	   Numeric value, lower RMS cutoff (in log10 scale).
    :param ref_sampling_rate:     Numeric value, expected sampling rate of audio signal in session_path.
    :param max_log_weight:	   Numeric value, maximum weight in powers of 10.
    :param min_log_weight:	   Numeric value, minimum weight in powers of 10.
    :param win_length:		   Numeric value, window length (frames).
    :param noise_sigma:           Numeric value, sigma of Gaussian noise added to the signal after
    				   applying the weights.
    				   
    :return: filtered_signal:     RMS-filtered audio signal in numpy array, same size as signal in session_path.
    :return: signal_filter:       Numpy array applied to the audio signal is session_path (piecewise multiplication).
    :return: rms_weights:	   Numpy array, the weights assidned to each frame.
    """

    # If "session_rms_log" is a 2D array, turn it into a 1D array. Raise error for other dims.
    if len(session_rms_log.shape) == 2:
        zero_dim = [i for i, x in enumerate(session_rms_log.shape) if x == 1][0]
        session_rms_log = session_rms_log.squeeze(zero_dim)
    if len(session_rms_log.shape) != 1:
        raise ValueError('Arg speech_signal should be either 1-dimensional or with shape (1, samples) OR (samples, 1)!')

    # Load wav
    signal, sr = lr.load(session_path, sr=None)
    # Sanity check
    if sr != ref_sampling_rate:
        raise ValueError('Unexpected sampling rate from file ' + session_path)

    # Call windowed weighting function on rms values
    rms_weights = windowed_weighting(session_rms_log,
                                     rms_threshold_high,
                                     rms_threshold_low,
                                     max_log_weight,
                                     min_log_weight,
                                     win_length=win_length)

    # Reconstruct weights array for the whole signal
    signal_filter = np.zeros(signal.shape)
    for frame in range(rms_weights.size):
        if frame != rms_weights.size - 1:
            frame_bounds = (session_sample_bounds[0, frame].astype(np.int64),
                            session_sample_bounds[0, frame + 1].astype(np.int64))
        elif frame == rms_weights.size - 1:
            frame_bounds = (session_sample_bounds[0, frame].astype(np.int64),
                            session_sample_bounds[1, frame].astype(np.int64))
        else:
            raise ValueError('Stg is horribly wrong...')
        signal_filter[frame_bounds[0]: frame_bounds[1]] = rms_weights[frame]

    # Apply median filter to array, to avoid cracking at sharp changes
    if medfilt_size:
        signal_filter = medfilt(signal_filter, kernel_size=medfilt_size)

    # Apply filter
    filtered_signal = np.multiply(signal, signal_filter)

    # Add Gaussian noise with "noise_sigma" st dev
    # Init np random generator
    rng = np.random.default_rng()
    noise_signal = rng.normal(loc=0.0, scale=noise_sigma, size=filtered_signal.shape)
    filtered_signal = filtered_signal + noise_signal

    return filtered_signal, signal_filter, rms_weights


def windowed_weighting(rms_log, rms_log_ceiling, rms_log_floor, max_log_weight, min_log_weight, win_length=11):
    """

    :param rms_log:
    :param rms_log_ceiling:
    :param rms_log_floor:
    :param max_log_weight:
    :param min_log_weight:
    :param win_length:
    :return:
    """
    win_lobe = np.floor(win_length/2).astype(np.int64)
    rms_padded = np.concatenate((np.zeros(win_lobe),
                                 rms_log,
                                 np.zeros(win_lobe)
                                 ))
    rms_weights = np.zeros(rms_padded.shape)
    for win_center in np.arange(win_lobe, rms_log.size + win_lobe):
        rms_weights[win_center] = weighting_fun(rms_padded[win_center - win_lobe: win_center + win_lobe + 1],
                                                rms_log_ceiling,
                                                rms_log_floor,
                                                max_log_weight,
                                                min_log_weight)

    return rms_weights[win_lobe: -win_lobe]


def weighting_fun(x_arr, x_max, x_min, y_max, y_min):
    """
    Weighting function to apply on each window of a signal (x_arr):
    (1) Values in window (x_arr) are averaged.
    (2) Weight is a linear function of the window average, but is constant above and below pre-specified values.
    The linear function determines the relative position of the window avg in a pre-specified range (x_max and x_min),
    and projects it into a specified range of weights (determined by y_may and y_min) (simple linear mapping within specified range).
    :param x_arr:  Numpy array holding signal.
    :param x_max:  Higher cutoff, above which the maximum weight is applied (log10 scale).
    :param x_min:  Lower cutoff, below which the minimum weight is applied (log10 scale).
    :param y_max:  Maximum weight (log10 scale).
    :param y_min:  Minimum weight (log10 scale).
    :return: weight: Weight to be applied to signal.
    """
    x_arr_mean = np.mean(x_arr)
    x_ratio = (x_arr_mean - x_min) / (x_max - x_min)
    if x_ratio > 1:
        x_ratio = 1
    elif x_ratio < 0:
        x_ratio = 0
    weight = x_ratio * (y_max - y_min) + y_min
    return 10**weight


def main():
    # Arguments for pair number, lab name and directory holding the outputs of ASR audio preprocessing
    # (outputs from audio_transcription_preproc.py)
    parser = argparse.ArgumentParser()
    parser.add_argument('pair_number', type=int, nargs=1,
                        help='Pair number, determines which audio (wav) files are selected for analysis.')
    parser.add_argument('lab', type=str, nargs=1,
                        help='Lab name, either "Mordor" or "Gondor".')
    parser.add_argument('--audio_dir', type=str, default=os.getcwd(), nargs='?',
                        help='Path to directory holding the results of ASR preprocessing (segment wav files, etc.).')
    parser.add_argument('--sessions', type=str, nargs='+', default=None,
                        help='Sessions to include, any or more of "BG1", "BG2", ..., "BGx", "freeConv". '
                             'Defaults to all sessions.')
    args = parser.parse_args()

    # Input checks and transforms
    if args.lab[0] not in ['Mordor', 'Gondor']:
        raise ValueError('Argument "lab" should either be "Mordor" or "Gondor"!')
    else:
        lab = args.lab[0]
    if not os.path.exists(args.audio_dir):
        raise NotADirectoryError('Argument "audio_dir" is not a directory!')
    else:
        audio_dir = args.audio_dir
    pair_number = args.pair_number[0]
    available_sessions = ['BG1', 'BG2', 'BG3', 'BG4', 'BG5', 'BG6', 'BG7', 'BG8', 'freeConv']
    if args.sessions:
        for session in args.sessions:
            if session not in available_sessions:
                raise ValueError('Values of input arg "sessions" should be "BGx" or "freeConv"!')

    print('\n\nCalled speech_rms_filtering with args:')
    print('Pair number: ' + str(pair_number))
    print('Lab: ' + lab)
    print('Audio directory: ' + audio_dir)
    if args.sessions:
        print('Sessions: ' + str(args.sessions))
    else:
        print('Sessions: All of them')

    ###########################
    #    HARDCODED PARAMS!    #
    sampling_rate = 44100
    low_rms_min_length_s = 0.25
    n_fft = 2048
    hop_length = np.floor(n_fft/2).astype(np.int32)
    max_log_weight = 0
    min_log_weight = -3
    win_length = 19
    gaussian_noise_sigma = 0.0005
    median_filter_size = 1999
    ###########################

    session_files = find_speech_files(pair=pair_number,
                                      lab=lab,
                                      audio_dir=audio_dir)
    print('\nFound ' + str(len(session_files)) + ' files:')
    for tmp in session_files:
        print(tmp)

    # If input arg "sessions" was provided, restrict the analysis to the sessions args.sessions
    if args.sessions:
        final_list = []
        for session in args.sessions:
            tmp_file = [f for f in session_files if session in f]
            if tmp_file:
                final_list.append(tmp_file[0])
        session_files = final_list
        print('\nRestricting analysis to files:')
        for tmp in session_files:
            print(tmp)

    print('\nEstimating per-frame RMS values of each audio...')
    log_rms_list, samples_list = get_log_rms(session_files,
                                             ref_sampling_rate=sampling_rate,
                                             n_fft=n_fft,
                                             hop_length=hop_length)
    print('Done.')

    print('\nPlotting RMS histogram across all frames in all audio files.')
    fig = plot_rms_hist(log_rms_list)

    print('\n\nPlease select the upper and lower cutoffs for RMS-weighted filtering:')
    rms_threshold_high, rms_threshold_low = get_thresh_input(fig)

    print('\nLooping through audio files, applying RMS-weighted filtering to each. '
          'Results are saved out to audio directory specified in the input args.')
    for idx, session in enumerate(session_files):
        session_rms_log = log_rms_list[idx]
        session_sample_bounds = samples_list[idx]

        print('\nWorking on ' + session + '...')
        filtered_signal, signal_filter, rms_weights = rms_weighting_filter(session,
                                                                           session_rms_log,
                                                                           session_sample_bounds,
                                                                           rms_threshold_high,
                                                                           rms_threshold_low,
                                                                           ref_sampling_rate=sampling_rate,
                                                                           max_log_weight=max_log_weight,
                                                                           min_log_weight=min_log_weight,
                                                                           win_length=win_length,
                                                                           noise_sigma=gaussian_noise_sigma,
                                                                           medfilt_size=median_filter_size
                                                                           )

        new_path = session[0:-4] + '_filtered.wav'
        sf.write(new_path, filtered_signal, samplerate=sampling_rate, subtype='PCM_16')
        print('Filtered audio saved out to ' + new_path)


if __name__ == '__main__':
    main()
