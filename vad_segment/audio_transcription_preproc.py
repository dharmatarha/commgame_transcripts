"""
Utility to segment a long audio recording into smaller ones at silent periods,
so it can be fed into BEAST2 ASR model for transcription.

Set up for CommGame audio recordings, with hardcoded file naming conventions, VAD parameters, folder handling, etc.

USAGE:
python audio_transcription_preproc.py 97 98 99 --audio_dir /media/gandalf/data_hdd/audio_transcription/data/raw/ [--use_filtered]

, where "97 98 99" are pair numbers in the CommGame dataset, and --audio_dir is the folder containing the .wav files
that belong to the pairs. If exactly two numbers are provided, they are treated as the first and last value in a range.

GENERAL LOGIC:
(1) Find silences in noise reduced long audio by using a VAD model (in this case, Silero-VAD set up in a local repo).
(2) Given the minimum and maximum length of segments, define speech segments between sufficiently long silences.
(3) Segment the not-noise-reduced, simply preprocessed, "raw" audio according to the segments defined on the
noise-reduced version of the audio.

DEPENDENCIES:
- Silero VAD (https://github.com/snakers4/silero-vad) in a local repo.
- Torchaudio and torch for audio file handling.

DETAILS:
- "Raw" input audio files should be already preprocessed in terms of (1) sampling rate repair,
(2) buffer underflow correction, and (3) stereo-to-mono conversion. The corrected files usually have the name
"pairPAIRNO_LABNAME_SESSION_repaired_mono.wav".
- "Noise reduced" input audio files are the "raw" files further processed by noise reduction algorithms. Their naming
follow the convention "pairPAIRNO_LABNAME_SESSION_repaired_mono_noisered.wav".
- "Raw" and "noise reduced" audio files should be arranged in subfolders named "raw" and "noise_reduced"
under --audio_dir.
- "Raw" and "noise_reduced" audio files must form pairs - there always must be both types for any session.
- This script is agnostic to SESSION, should work with any.
- All audio files for the given pair numbers in --audio_dir/raw and --audio_dir/noise_reduced are processed in a loop.
- For each pair number, a result folder is created under --audio_dir.
- For each long audio file, all results are saved out into a folder named after the audio file, created
under the pair folder.
- Silero-VAD and the downstream BEAST2 both require 16 kHz sampling rate, so audio is resampled to 16k. We expect that
the original sampling rate is higher (usually 44.1 kHz).
- Silero-VAD outputs both locations of speech (in samples) and probability of speech for each sample. These are
saved out as "AUDIO_BASE_FILENAME_vad.json" and "AUDIO_BASE_FILENAME_vad_probs.npy".
- Speech locations are also turned into a numpy array of zeros and ones, where zeros mark silence and ones mark speech.
The length of the array equals the length of the resampled long audio. The array is saved out as
"AUDIO_BASE_FILENAME_vad_speech.npy".
- Short segments have hardcoded minimum-maximum length values. Cut points are at the start and end of the speech
segments between sufficiently long silences (see param min_silence_s), defined with padding (param padding_s).
Speech segment boundaries (in samples) are saved out as a numpy array into
"AUDIO_BASE_FILENAME_segmentation_samples.npz".
- For each long audio, a dataset-descriptor .json file is created for batch transcription with BEAST2,
named "AUDIO_BASE_FILENAME_data.json".
- For each long audio, a BEAST2 ASR hyperparameter and model spec .yaml file is generated for batch transcription
with BEAST2, named "AUDIO_BASE_FILENAME_hparams.yaml".

"""


import argparse
import torchaudio
import torch
import json
from time import time
import os
import numpy as np
from glob import glob
import ruamel.yaml


def find_audio(pair_number, audio_path, filtered=False):
    """
    Find all "raw" and "noise_reduced" audio files for a given pair number in the directory specified by "audio_path".
    Dir "audio_path" must contain these files in subdirs "/raw" and "/noise_reduced", respectively.
    Audio files conform to these formats, for "raw" and "noise_reduced":
        RAW:            "pair[PAIR_NO]_[Mordor|Gondor]_[freeConv|BGx]_repaired_mono.wav"
        NOISE_REDUCED:  "pair[PAIR_NO]_[Mordor|Gondor]_[freeConv|BGx]_repaired_mono_noisered.wav" OR
                        "pair[PAIR_NO]_[Mordor|Gondor]_[freeConv|BGx]_repaired_mono_noisered_filtered.wav"
    Input arg "filtered" determines the version of "noise_reduced" files to list.

    :param pair_number:   List of integers, pair numbers.
    :param audio_path:    Str, path to directory containing audio files.
    :param filtered:      Bool, flag for using filtered audio for "noise_reduced" files
                          (that is, wav files with "_filtered.wav" ending).
    :return: audio_files: List of lists. Each element of audio_files is a list of tuples, one list for each pair number.
                          Each tuple contains the "raw" and "noise_reduced" versions of audio files found for the pair.
    """
    audio_files = []
    for current_pair in pair_number:
        # first find all "raw" audio files for a given pair
        raw_format = 'pair' + str(current_pair) + '_*dor_*_repaired_mono.wav'
        raw_files_list = glob(os.path.join(audio_path, 'raw', raw_format))
        # loop through each raw audio, find corresponding "noise_reduced" audio, and form tuples
        pair_list = []
        for raw_file in raw_files_list:
            raw_filename = os.path.split(raw_file)[1]
            if filtered:
                noisered_format = raw_filename[:-4] + '_noisered_filtered.wav'
            else:
                noisered_format = raw_filename[:-4] + '_noisered.wav'
            noisered_file = os.path.join(audio_path, 'noise_reduced', noisered_format)
            if os.path.exists(noisered_file):
                pair_list.append((raw_file, noisered_file))
            else:
                raise FileNotFoundError('Could not find "noise_reduced" file: ' + noisered_file)

        # append pair-specific list to output list
        audio_files.append(pair_list)

    return audio_files


def prepare_vad(vad_repo_location='/home/gandalf/beast2/snakers4_silero-vad_master'):
    """
    Function to load silero-vad model and its utilities.
    See https://github.com/snakers4/silero-vad

    Only get_speech_timestamps() is returned from utils!!!
    Sets torch to CPU device and 1 thread!!!

    :param vad_repo_location: Str, path to silero-vad repo locally. Defaults to
                              '/home/gandalf/beast2/snakers4_silero-vad_master'.
    :return: vad_model:       VAD model object.
    :return: get_speech_timestamps: Utility function for querying speech segments in a given audio waveform.
    """
    # torch settings
    torch.set_num_threads(1)
    device = torch.device('cpu')
    # load silero_vad model
    print('\nLoading VAD model')
    model_load_start = time()
    vad_model, vad_utils = torch.hub.load(repo_or_dir=vad_repo_location,
                                          source='local',
                                          model='silero_vad',
                                          onnx=False,
                                          trust_repo=True)
    vad_model.to(device)
    get_speech_timestamps = vad_utils[0]
    print('Model loaded, took', round(time() - model_load_start, 3), 'seconds')

    return vad_model, get_speech_timestamps


def wav_resample(wav_file, resampling_rate=16000, device='cpu'):
    """
    Loads and resamples .wav file to "resampling_rate" Hz. Uses torchaudio, hence the option "device".

    Only intended for and tested with mono audio!!!
    Only for audio sampled at a higher frequency than resampling_rate!!!

    :param wav_file:        Str, path to .wav file.
    :param resampling_rate: Int, new sampling rate for resampling.
    :param device:          Pytorch device for resampled audio waveform
    :return: resampled_waveform: torch.tensor of the resampled audio.
    """
    print('Loading wav file ' + wav_file)
    metadata = torchaudio.info(wav_file)
    print(metadata)
    waveform, sampling_rate = torchaudio.load(wav_file)
    print('Resampling to ' + str(resampling_rate) + ' Hz')
    resampling_start = time()
    resampled_waveform = torchaudio.functional.resample(waveform,
                                                        sampling_rate,
                                                        resampling_rate,
                                                        lowpass_filter_width=64,
                                                        rolloff=0.95,
                                                        resampling_method='sinc_interp_kaiser',
                                                        beta=14.77)
    resampled_waveform.to(device)
    print('Resampling done, took', round(time() - resampling_start, 3), 'seconds')

    return resampled_waveform


def get_probabilities(waveform, vad_model, window_size_samples=512, sampling_rate=16000):
    """
    :param waveform:            torch.tensor containing audio waveform data with shape [1, SAMPLES]
    :param vad_model:           Loaded silero-vad model object
    :param window_size_samples: Int, window size for querying probabilities, in samples
    :param sampling_rate:       Int, sampling rate of waveform in Hz
    :return: speech_probs_nparray: Numpy array, contains speech probabilities, for each sample in waveform.
    """
    print('Querying per-frame probabilities from VAD model')
    prob_start = time()
    speech_probs = []
    audio_length_samples = waveform.shape[1]
    tmp_audio = waveform.clone().detach()
    tmp_audio = tmp_audio.squeeze(0)
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = tmp_audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_prob = vad_model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)
    speech_probs_nparray = np.array(speech_probs, dtype=np.float32)
    print('VAD model finished with probabilities, took', round(time()-prob_start, 3), 'seconds')

    return speech_probs_nparray


def get_audio_cut_points(speech_signal, sampling_rate=16000, min_length_s=2, max_length_s=15, segment_extend_s=3):
    """
    Function to find cut (segmentation) points of long speech recording that correspond to the start and end of long
    silent periods in the signal.
    The length of the resulting segments is defined by args min_length_s and max_length_s. Returns the speech segment
    start and end points in samples.

    :param speech_signal:      Numpy array, float, either 1-dimensional or with a shape of (1, samples) OR (samples, 1).
                               Silence is marked by zero values in the signal, everything else is speech.
    :param sampling_rate:      Int, sampling rate of speech_signal in Hz.
    :param min_length_s:       Int, minimum length in seconds between cut points (=minimum length of segments if
                               speech_signal is segmented according to the output of this function).
    :param max_length_s:       Int, maximum length in seconds between cut points (=maximum length of segments if
                               speech_signal is segmented according to the output of this function).
    :param segment_extend_s:   Int, the length of window extension allowed if there is no silence (segment with zeros)
                               in a segment within the boundaries provided by min_length_s and max_length_s. Provided
                               in seconds. In such cases, the window is extended by "segment_extend_s" seconds.
    :return: segment_starts:   List of speech segment starts, in samples.
    :return: segment_ends:     List of speech segment ends, in samples.
    """
    # Squeeze speech_signal if it has an empty dimension, we need 1-dimensional signal
    if len(speech_signal.shape) == 2:
        zero_dim = [i for i, x in enumerate(speech_signal.shape) if x == 1][0]
        speech_signal = speech_signal.squeeze(zero_dim)
    if len(speech_signal.shape) != 1:
        raise ValueError('Arg speech_signal should be either 1-dimensional or with shape (1, samples) OR (samples, 1)!')

    # Get constants for windows
    min_length_samples = int(np.floor(min_length_s * sampling_rate))
    max_length_samples = int(np.floor(max_length_s * sampling_rate))
    seg_extend_samples = int(np.floor(segment_extend_s * sampling_rate))

    # Lists for holding speech segment bounds
    segment_starts = []
    segment_ends = []

    # Get beginning of first window: first speech sample minus 1
    speech_sample_indices = np.asarray(speech_signal != 0).nonzero()[0]
    if speech_sample_indices.size == 0:
        raise ValueError('There are only zeros in speech_signal! (No speech?)')
    segment_start_idx = max(int(speech_sample_indices[0]-1), 0)

    # Define speech segment bounds in a loop
    max_idx = speech_signal.size - 1
    end_of_signal_flag = False
    while not end_of_signal_flag:

        # Get end of current speech segment: given the minimum and maximum length,
        # get the start of the largest silence plus 1
        cut_window_start = segment_start_idx + min_length_samples
        cut_window_end = segment_start_idx + max_length_samples

        # If the window for silences is longer than we have data left, mark the last speech sample in the signal as
        # the end of the current speech segment and set the flag for breaking out the while loop
        if cut_window_end >= max_idx:
            segment_end_idx = speech_sample_indices[-1]
            segment_starts.append(segment_start_idx)
            segment_ends.append(segment_end_idx)
            end_of_signal_flag = True

        # Otherwise there is still data left in the signal. In this case, find the longest silence in the
        #  appropriate window and derive the segment end and the start of the next segment.
        else:
            tmp_signal = speech_signal[cut_window_start: cut_window_end]

            # In case there is no silence (zeros) in tmp_signal, apply window extension if allowed,
            # raise error otherwise
            if tmp_signal.size == np.sum(tmp_signal):
                if segment_extend_s > 0:
                    cut_window_end = min(cut_window_end + seg_extend_samples, max_idx-1)
                    tmp_signal = speech_signal[cut_window_start: cut_window_end]
                    # If there is still no silence, even in extended window, raise error
                    if tmp_signal.size == np.sum(tmp_signal):
                        raise ValueError('There is silence (zero run) in the signal between samples '
                                         + str(cut_window_start) + ' and ' + str(cut_window_end) + '!')
                else:
                    raise ValueError('There is silence (zero run) in the signal between samples '
                                     + str(cut_window_start) + ' and ' + str(cut_window_end) + '!')

            # Get all zero runs in window
            zero_ranges = zero_runs(tmp_signal)
            # Find longest zero run
            max_range_length = np.max(np.diff(zero_ranges))
            max_range_idx = np.argmax(np.diff(zero_ranges))
            range_bounds = zero_ranges[max_range_idx, :]

            # If the maximum length zero run has a length longer than 1, define its start plus 1 as the end of
            # the current speech segment
            if max_range_length < 2:
                raise ValueError('Could not find a zero run longer than 1 for ending a speech segment!')
            else:
                segment_end_idx = range_bounds[0] + cut_window_start + 1
                segment_starts.append(segment_start_idx)
                segment_ends.append(segment_end_idx)

                # Start of next segment is the end of longest silence in window minus 1
                segment_start_idx = range_bounds[1] + cut_window_start - 1

    return segment_starts, segment_ends


def get_audio_cut_points_only_speech(speech_signal, sampling_rate=16000, min_length_s=0.5, max_length_s=20,
                                     min_silence_s=0.75, padding_s=0.1):
    """
    Alternative function to find shorter segments of speech in a long speech recording. The length of the resulting
    segments is defined by args min_length_s and max_length_s. Returns the speech segment start and end points
    in samples.

    Logic: Start at the first frames with speech, extend the end of the segment until a silent part with at
    least min_silence_s length is reached (or terminate at min_length_s from start if silence is too soon).
    Terminate the segment at the silent part. Next segment starts at the next speech frame. Repeat.

    Throws an error if there is no suitable silence to terminate speech segment on between
    segment start and (segment start + max_length_s).

    :param speech_signal:      Numpy array, float, either 1-dimensional or with a shape of (1, samples) OR (samples, 1).
                               Silence is marked by zero values in the signal, everything else is speech.
    :param sampling_rate:      Int, sampling rate of speech_signal in Hz. Defaults to 16000.
    :param min_length_s:       Int / float, minimum length in seconds between cut points (=minimum length of segments if
                               speech_signal is segmented according to the output of this function). Defaults to 1.
    :param max_length_s:       Int / float, maximum length in seconds between cut points (=maximum length of segments if
                               speech_signal is segmented according to the output of this function). Defaults to 20.
    :param min_silence_s:      Int / float,
    :param padding_s:          Int / float,
    :return: segment_starts:   List of speech segment starts, in samples.
    :return: segment_ends:     List of speech segment ends, in samples.
    """
    # Squeeze speech_signal if it has an empty dimension, we need 1-dimensional signal
    if len(speech_signal.shape) == 2:
        zero_dim = [i for i, x in enumerate(speech_signal.shape) if x == 1][0]
        speech_signal = speech_signal.squeeze(zero_dim)
    if len(speech_signal.shape) != 1:
        raise ValueError('Arg speech_signal should be either 1-dimensional or with shape (1, samples) OR (samples, 1)!')

    # Get constants for windows
    min_length_samples = int(np.floor(min_length_s * sampling_rate))
    max_length_samples = int(np.floor(max_length_s * sampling_rate))
    min_silence_samples = int(np.floor(min_silence_s * sampling_rate))
    padding_samples = int(np.floor(padding_s * sampling_rate))

    # Lists for holding speech segment bounds
    segment_starts = []
    segment_ends = []

    # Get beginning of first window: first speech sample minus padding
    speech_sample_indices = np.asarray(speech_signal != 0).nonzero()[0]
    if speech_sample_indices.size == 0:
        raise ValueError('There are only zeros in speech_signal! (No speech?)')
    segment_start_idx = max(int(speech_sample_indices[0] - padding_samples), 0)

    # get zero runs (silent parts) in the whole signal
    zero_ranges = zero_runs(speech_signal)
    # add a column with the length of each zero run (silent segment)
    zero_ranges = np.hstack((zero_ranges, np.diff(zero_ranges)))

    end_of_signal_flag = False
    last_segment_to_ending_flag = False
    segment_end_idx = None
    while not end_of_signal_flag:

        # If there is less than min_length_samples left from the signal, abort segmentation,
        # remaining part of the signal is to be attached to the previous segment.
        if segment_start_idx + min_length_samples >= speech_signal.size - 1:
            last_segment_to_ending_flag = True
            end_of_signal_flag = True

        # Else check for zero runs following the segment start that meet the following requirements:
        # (1) do not start later than max_length_samples; (2) their ending is at soonest at min_length_samples;
        # and (3) their length is at least min_silence_samples.
        # segment_end_ranges will be an np.array holding the row indexes of zero_ranges passing the conditions
        else:
            segment_end_ranges = np.where(np.all((zero_ranges[:, 0] > segment_start_idx,
                                                 zero_ranges[:, 0] < segment_start_idx + max_length_samples,
                                                 zero_ranges[:, 1] > segment_start_idx + min_length_samples,
                                                 zero_ranges[:, 2] >= min_silence_samples), axis=0))[0]

            # if there is at least one suitable silent part (zero run), select the first one for the end of the segment
            if segment_end_ranges.size > 0:
                segment_end_idx = zero_ranges[segment_end_ranges[0], 0] - 1 + padding_samples  # -1 so that we get the index of the last speech sample

            # If there is none, that might be because we are relatively close to the end of the whole signal
            # and there is only speech left. In that case, segment end is at the end of the signal.
            elif segment_start_idx + max_length_samples >= speech_signal.size:
                segment_end_idx = speech_signal.size - 1
                end_of_signal_flag = True

            # In all other cases, it seems rather strange that we could not find a proper zero run for segmenting.
            # Raise an error because stg might have gone terribly wrong
            else:
                raise ValueError('Could not find suitable end for segment starting at ' + str(segment_start_idx) + '!!!')

        # Append segment start and end lists
        if not last_segment_to_ending_flag:
            segment_starts.append(segment_start_idx)
            segment_ends.append(segment_end_idx)
        else:
            segment_ends[-1] = speech_signal.size - 1

        # If we are not at the end of the signal, find the start of the next segment and go on with the loop
        if not end_of_signal_flag:
            # Starting from segment end, find the next zero run ending
            segment_end_ranges = np.where(zero_ranges[:, 1] > segment_end_idx)[0]
            # If it exists, and it is not at the end of the signal, the next segment start will be
            # at the ending of the zero run minus padding
            if segment_end_ranges.size > 0:
                zero_run_end_idx = zero_ranges[segment_end_ranges[0], 1]
                if zero_run_end_idx != speech_signal.size - 1:
                    segment_start_idx = zero_run_end_idx + 1 - padding_samples
                else:
                    end_of_signal_flag = True
            # If the logic of segmentation is working as intended, and all arguments are sane,
            # we should never come to this scenario here. So raise an error if we do...
            else:
                raise ValueError('Something went very wrong. Think through the input arguments!'
                                 'If this error persists, stg is broken!!!')

    return segment_starts, segment_ends


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


def trim_zeros_from_other(arr1, arr2):
    """
    Utility for deleting values from arr1 where arr2 has leading of trailing zeros. In other words, if arr2 has
    leading or trailing zeros, we trim arr1 using the leading / trailing zeros of arr2 as the indices for deleting
    arr1 values.

    Input arrays arr1 and arr2 must have the same shape!!!
    Works only on arrays with shape (1, N)!!!

    :param arr1:  Numpy array shaped (1, N).
    :param arr2:  Numpy array shaped (1, N). Same shape as arr1!
    :return: arr_trimmed: Numpy array, after trimming arr1 values corresponding to leading/trailing zeros in arr2.
    """
    # Input checks
    if arr1.shape != arr2.shape:
        raise ValueError('Input arrays must have the same shape!')
    if arr1.shape[0] != 1:
        raise ValueError('Input arrays must have shape (1, N)!')

    # Do not alter original input array in place!
    arr_trimmed = arr1.copy()

    # Get indices for all non-zero values in arr2
    nonzero_indices = np.asarray(arr2 != 0).nonzero()[1]

    # special case: if there are no non-zero elements, return a (1, 1) shaped array with a zero value
    if nonzero_indices.size == 0:
        arr_trimmed = np.asarray([[0]])

    else:

        # If there are trailing zeros, delete corresponding part from arr1
        # Important to start with the end, otherwise indices get screwed up
        if nonzero_indices[-1] != (arr2.shape[1]-1):
            indices_to_delete = np.arange(nonzero_indices[-1]+1, arr2.shape[1])
            arr_trimmed = np.delete(arr_trimmed, indices_to_delete, axis=1)
        # if there are leading zeros, delete corresponding part from arr1
        if nonzero_indices[0] != 0:
            indices_to_delete = np.arange(0, nonzero_indices[0])
            arr_trimmed = np.delete(arr_trimmed, indices_to_delete, axis=1)

    return arr_trimmed


def segment_audio(long_audio, segment_starts, segment_ends, base_path,
                  sampling_rate=16000,
                  audio_format='wav',
                  encoding='PCM_S',
                  bits_per_sample=16):
    """
    Function to cut long audio into pieces, at given "cut points".
    Each segment is saved out into a separate audio file (by default, a 16bit wav file),
    with the filename derived from "base_path" (base_path + '_segmentX.FORMAT').

    Audio must be numpy array or torch tensor with shape (1, samples), that is, single channel audio only!!!

    :param long_audio:      Numpy array or torch.tensor with shape (1, samples), corresponding to a
                            single channel of audio signal.
    :param segment_starts:  List or 1D array of integers, used as sample indices for segmentation starts.
                            Must be in monotonically increasing order!
    :param segment_ends:    List or 1D array of integers, used as sample indices for segmentation ends.
                            Must be in monotonically increasing order!
    :param base_path:       Str, file path for output wav files. Must contain the base filename as well,
                            as output wav files are named base_path + '_segmentX.FORMAT'.
    :param sampling_rate:   Sampling rate in Hz, passed to torchaudio.save. Defaults to 16000.
    :param audio_format:    Audio file format, passed to torchaudio.save. Defaults to 'wav'.
    :param encoding:        Audio encoding, passed to torchaudio.save. Defaults to 'PCM_S'.
    :param bits_per_sample: Bits per sample used with the encoding, passed to torchaudio.save. Defaults to 16.
    :return: all_segment_paths:   List of file paths, contains the path to all saved-out segments.
    """

    # Check if audio is torch tensor, if not, try to cast it to tensor,
    # and hope that the error is informative if it fails
    if type(long_audio).__name__ != 'Tensor':
        long_audio = torch.tensor(long_audio)
    # Check audio shape, should be (1, samples) or (samples, 1)
    if len(long_audio.shape) != 2:
        raise ValueError('Arg long_audio should be 2D tensor!')
    if np.min(long_audio.shape) != 1:
        raise ValueError('Arg long_audio should have shape (1, samples) or (samples, 1)!')
    # Normalize shape into (1, samples)
    if long_audio.shape[1] == 1:
        long_audio = long_audio.T

    # Sanity checks on segment_starts and segment_ends
    # Are they monotonic increasing? Are they the same size?
    if not all([True for c in np.diff(segment_starts) if c > 0]):
        raise ValueError('Arg segment_starts should contain monotone increasing values (sample indices)!')
    if not all([True for c in np.diff(segment_ends) if c > 0]):
        raise ValueError('Arg segment_ends should contain monotone increasing values (sample indices)!')
    if segment_starts[-1] >= np.max(long_audio.shape):
        raise ValueError('Last value in segment_starts is too high to be an index for long_audio!')
    if segment_ends[-1] >= np.max(long_audio.shape):
        raise ValueError('Last value in segment_ends is too high to be an index for long_audio!')
    if len(segment_starts) != len(segment_ends):
        raise ValueError('Args segment_starts and segment_ends should have same size!')

    # Loop through segments, pass format-related args to torchaudio.save,
    # save out each segment
    all_segment_paths = []
    segment_idx = 0
    for start_idx, end_idx in zip(segment_starts, segment_ends):
        # Get segment from long_audio
        segment = long_audio[:, start_idx: end_idx]
        # File name handling
        segment_filepath = base_path + '_segment' + str(segment_idx)
        all_segment_paths.append(segment_filepath)
        torchaudio.save(segment_filepath, segment,
                        sample_rate=sampling_rate,
                        channels_first=True,
                        format=audio_format,
                        encoding=encoding,
                        bits_per_sample=bits_per_sample)
        segment_idx = segment_idx + 1

    return all_segment_paths


def json_for_beast(all_segment_paths, segmentation_path, sampling_rate, file_basename):
    """
    Function to generate a data-descriptor json file for batch inference (transcription) using beast2 ASR model.
    Beast2 implementation expects a json that contains the path and length of each .wav file.

    :param all_segment_paths:    List of audio segment (.wav file) paths
    :param segmentation_path:    Str, path to .npz file containing the segmentation points
                                 (see "get_audio_cut_points" function)
    :param sampling_rate:        Numeric value, sampling rate in Hz.
    :param file_basename:        Str, the filename prefix shared by all result files for a given audio recording.
    :return: json_path:          Str, path to the resulting .json file.
    """
    # Load cut_points file and get length of each segment
    tmp = np.load(segmentation_path)
    segment_starts = tmp['segment_starts']
    segment_ends = tmp['segment_ends']
    segment_lengths = np.asarray([e-s for s, e in zip(segment_starts, segment_ends)])
    segment_lengths_s = segment_lengths/sampling_rate
    # Create a dict of dicts for the json
    data_dict = {}
    for segment, segment_len in zip(all_segment_paths, segment_lengths_s):
        segment_filename = os.path.split(segment)[1]
        data_dict[segment_filename] = {'wav': segment, 'length': segment_len, 'words': 'x'}
    # Save dict as json
    json_path = os.path.join(os.path.split(all_segment_paths[0])[0], file_basename + '_data.json')
    with open(json_path, 'w') as file_out:
        json.dump(data_dict, file_out)
    print('JSON created at:')
    print(json_path)

    return json_path


def define_yaml(base_yaml_path, file_dir, file_basename, asr_json_path):
    """
    Function to adjust a "base" .yaml file for BEAST2 ASR model with specific data and output paths.

    :param base_yaml_path:  String, path to "base" .yaml file (e.g. default debug_with_lm.yaml from BEAST2 model repo).
    :param file_dir:        String, path to folder containing the .wav files for waiting for transcription.
    :param file_basename:   String, basename of original long speech recording.
    :param asr_json_path:   String, path to data descriptor .json file necessary for BEAST2 transcription.
    :return: yaml_path:     String, path to data-specific .yaml file.
    """
    yaml = ruamel.yaml.YAML()
    with open(base_yaml_path) as file_in:
        data = yaml.load(file_in)
    # Add values to keys 'data_folder', 'test_json' and 'wer_file'
    data['data_folder'] = file_dir
    data['test_json'] = asr_json_path
    data['wer_file'] = os.path.join(file_dir, file_basename + '_beast2_results.txt')
    # Dump yaml
    yaml_path = os.path.join(file_dir, file_basename + '_hparams.yaml')
    with open(yaml_path, 'w') as file_out:
        yaml.dump(data, file_out)

    return yaml_path


def main():
    # Input argument handling
    # One mandatory and one optional argument, defining pair number(s) and the path to the audio (wav) files.
    # Arg args.pair_numbers is a list with one or more elements while args.audio_dir is string (path to dir).
    parser = argparse.ArgumentParser()
    parser.add_argument('pair_numbers', type=int, nargs='+',
                        help='Pair numbers, determines which audio (wav) files are selected for analysis.'
                             'If exactly two numbers are provided, a range of pair numbers is defined by treating '
                             'the two numbers as the first and last pair numbers.')
    parser.add_argument('--audio_dir', type=str, default=os.getcwd(), nargs='?',
                        help='Path to directory holding the audio (wav) files, '
                             'with subdirectories "/raw", "/noise_reduced", and "/asr".')
    parser.add_argument('--use_filtered', action='store_true',
                        help='Flag for using rms-filtered audio as input, that is, '
                             'audio files with "_filtered.wav" ending.')
    args = parser.parse_args()

    # Checks for necessary directories
    if not os.path.exists(args.audio_dir):
        raise NotADirectoryError('Input arg --audio_dir is not a valid path!')
    if (not os.path.exists(os.path.join(args.audio_dir, 'raw'))
            or not os.path.exists(os.path.join(args.audio_dir, 'noise_reduced'))
            or not os.path.exists(os.path.join(args.audio_dir, 'asr'))):
        raise NotADirectoryError('Input arg --audio_dir does not have subdirectories "/raw", '
                                 '"/noise_reduced", or "/asr"!')

    # If there are two pair numbers, expand it into a range
    if len(args.pair_numbers) == 2:
        pair_numbers = list(range(args.pair_numbers[0], args.pair_numbers[1]+1))
    else:
        pair_numbers = args.pair_numbers

    ########## HARDCODED VARS #############
    # THERE ARE FURTHER HARDCODED PARAMS IN FUNCTION CALLS
    # !!! VAD AND BEAST2 MODELS ONLY WORK WITH 16K !!!
    resampling_rate = 16000
    # !!! BASE HYPERPARAMETER AND MODEL SPECS YAML FILE PATH !!!
    base_yaml_path = '/home/gandalf/beast2/decode_with_lm_base.yaml'
    #######################################

    print('\nCalled audio_preproc with args:')
    print('Pair numbers: ' + str(pair_numbers))
    print('Audio directory: ' + args.audio_dir)
    print('Flag for filtered audio input: ' + str(args.use_filtered))
    print('\nLooking for relevant audio files...')

    # Find all files for the supplied pair number(s) and directory.
    # Returned var is a list of lists, with each list containing tuples of the relevant files (raw and noise_reduced)
    # for the corresponding element in "args.pair_numbers".
    audio_files = find_audio(pair_numbers, args.audio_dir, filtered=args.use_filtered)

    total_files = sum([len(i) for i in audio_files])
    print('A total of ' + str(total_files) + ' tuples of audio files were found across pair numbers.')

    # Load VAD model and its utility functions
    vad_model, get_speech_timestamps = prepare_vad()

    # Loops over pair number and the corresponding audio files
    for pair_idx, current_pair in enumerate(pair_numbers):
        print('\n\nStarting work on pair ' + str(current_pair))
        print('\nFiles for pair ' + str(current_pair) + ':')
        for pair_wav_tuple in audio_files[pair_idx]:
            print(pair_wav_tuple)

        # Make pair-specific directory if it does not exist yet
        pair_dir = os.path.join(args.audio_dir, 'asr', 'pair' + str(current_pair))
        if not os.path.exists(pair_dir):
            os.mkdir(pair_dir)
            print('Created pair-specific dir at ' + pair_dir)

        # Loop through audio files
        for pair_wav_tuple in audio_files[pair_idx]:
            print('\nWorking on wav files:')
            print(pair_wav_tuple[0])
            print(pair_wav_tuple[1])
            # File basename is derived from noise_reduced files
            file_basename = os.path.basename(pair_wav_tuple[1]).split('.')[0]
            # Make file-specific directory if it does not exist yet
            file_dir = os.path.join(pair_dir, file_basename)
            if not os.path.exists(file_dir):
                os.mkdir(file_dir)
                print('Created file-specific dir at ' + file_dir)

            # load and resample both wav files
            resampled_raw_waveform = wav_resample(pair_wav_tuple[0], resampling_rate=resampling_rate)
            resampled_noisered_waveform = wav_resample(pair_wav_tuple[1], resampling_rate=resampling_rate)

            # get speech timestamps from full audio file
            print('Querying timestamps for resampled waveform')
            vad_start = time()
            speech_timestamps = get_speech_timestamps(resampled_noisered_waveform,
                                                      vad_model,
                                                      sampling_rate=resampling_rate,
                                                      threshold=0.95,
                                                      min_speech_duration_ms=200,
                                                      min_silence_duration_ms=100,
                                                      window_size_samples=512,
                                                      speech_pad_ms=30)
            print('VAD model finished, took', round(time() - vad_start, 3), 'seconds')

            # Save speech timestamps as json
            vad_json_filename = file_basename + '_vad.json'
            vad_json_path = os.path.join(file_dir, vad_json_filename)
            with open(vad_json_path, 'w') as json_out:
                json.dump(speech_timestamps, json_out)
            print('Speech timestamps saved out into .json file:')
            print(vad_json_path)

            # Turn speech timestamps into a numpy array and save it out
            speech_signal = np.zeros(resampled_noisered_waveform.shape)
            for frame in speech_timestamps:
                speech_signal[0, frame['start'] - 1: frame['end']] = 1
            speech_signal_filename = file_basename + '_vad_speech.npy'
            speech_signal_path = os.path.join(file_dir, speech_signal_filename)
            np.save(speech_signal_path, speech_signal, allow_pickle=False)
            print('Speech signal timeseries saved out into .npy file:')
            print(speech_signal_path)

            # Get speech probabilities for each frame
            # Returned "speech_probs" is a numpy array
            speech_probs = get_probabilities(resampled_noisered_waveform,
                                             vad_model,
                                             window_size_samples=512,
                                             sampling_rate=resampling_rate)

            # Save probabilities as .npy file
            speech_probs_filename = file_basename + '_vad_probs.npy'
            speech_probs_path = os.path.join(file_dir, speech_probs_filename)
            np.save(speech_probs_path, speech_probs, allow_pickle=False)
            print('Probabilities saved out into .npy file:')
            print(speech_probs_path)

            # Get speech segment boundaries (starts and ends) for segmenting the long recording
            print('Looking for ideal cuts with regards to silent segments...')
            segment_starts, segment_ends = get_audio_cut_points_only_speech(speech_signal,
                                                                            sampling_rate=16000,
                                                                            min_length_s=0.5,
                                                                            max_length_s=30,
                                                                            min_silence_s=0.5,
                                                                            padding_s=0.3)
            # Save out segment start and end lists as np arrays, in .npz file
            segmentation_filename = file_basename + '_segmentation_samples.npz'
            segmentation_path = os.path.join(file_dir, segmentation_filename)
            np.savez(segmentation_path,
                     segment_starts=np.array(segment_starts),
                     segment_ends=np.array(segment_ends),
                     allow_pickle=False)
            print('Cut points (segment start and end points in samples) saved out into .npz file:')
            print(segmentation_path)

            # Cut "RAW" (!) long audio into segments, according to segmentation starts and ends
            print('Segmenting resampled audio and saving out each segment...')
            all_segment_paths = segment_audio(resampled_raw_waveform,
                                              segment_starts,
                                              segment_ends,
                                              os.path.join(file_dir, file_basename),
                                              sampling_rate=16000,
                                              audio_format='wav',
                                              encoding='PCM_S',
                                              bits_per_sample=16)
            print('Saved out ' + str(len(all_segment_paths)) + ' segments.')

            # Get a data json used for BEAST2 transcription
            print('Generating data descriptor .json for BEAST2 ASR...')
            asr_json_path = json_for_beast(all_segment_paths,
                                           segmentation_path,
                                           resampling_rate,
                                           file_basename)
            print('Saved out BEAST2 data .json:')
            print(asr_json_path)

            # Create custom hyperparameter .yaml file for audio that specifies correct data and output paths
            print('Generating hyperparameter and model spec .yaml file for BEAST2 ASR...')
            asr_yaml_path = define_yaml(base_yaml_path,
                                        file_dir,
                                        file_basename,
                                        asr_json_path)
            print('Saved out BEAST2 ASR .yaml:')
            print(asr_yaml_path)

        print('\nFinished with all files for pair ' + str(current_pair) + '!')
    print('\nFinished with all pairs!')


if __name__ == '__main__':
    main()
