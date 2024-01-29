"""
Utility for generating new audio segments form manually fixed transcripts in the CommGame data set.

USAGE:
python srt_to_segments.py --data_dir DATA_DIR

where DATA_DIR should point to the central data folder with the audio and transcription data, with subdirectories
"asr", "raw", "noise_reduced", and "fixed".

The script lists the fixed transcripts under "DATA_DIR/fixed" and segments the corresponding raw audio files according
to the new srt file. These new audio segments are stored under
"DATA_DIR/asr/pair[PAIRNO]/pair[PAIRNO]_[LAB]_[SESSION]_fixed_segmentation/"
"""

import srt
import torchaudio
import torch
import os
import numpy as np
import argparse


def info_for_segmentation(data_dir):
    """
    Function to look through all ""data_dir/fixed" directories, and collect info about the manually fixed transcripts.
    Specifically, for each manually fixed transcript, the function
    (1) finds the corresponding raw audio file under "data_dir/raw",
    (2) creates a directory for the new audio segments based on the fixed transcript under "data_dir/asr/pair_dir/",
    (3) and determines the basename of each new audio segment file.

    These are all returned in a list of tuples, where each tuple consists of the path of the manually fixed transcript
    and the above three corresponding strings (raw audio file path, path to directory for new segments, and
    segment basename).

    :param data_dir: Path to data directory holding audio data and audio segments for transcripts. Must have subdirs
                     "asr" and "raw"
    :return: segmentation_list: List of tuples. See the docstring for the content of each tuple.
    """

    segmentation_list = []

    fixed_dir = os.path.join(data_dir, 'fixed')
    raw_dir = os.path.join(data_dir, 'raw')
    asr_dir = os.path.join(data_dir, 'asr')

    fixed_pairs = os.listdir(os.path.join(data_dir, 'fixed'))

    for pair_dir in fixed_pairs:
        file_list = os.listdir(os.path.join(fixed_dir, pair_dir))
        if file_list:

            for file_path in file_list:

                # Derive pair number, lab and session for given file
                file_parts = file_path.split('_')
                pair_no = file_parts[0][4:]
                lab = file_parts[1]
                session = file_parts[2]

                # Get full path of srt file
                srt_path = os.path.join(fixed_dir, pair_dir, file_path)

                # Get path of corresponding raw audio
                audio_filename = '_'.join(['pair' + str(pair_no),
                                          lab,
                                          session,
                                          'repaired_mono.wav'])
                audio_path = os.path.join(raw_dir, audio_filename)
                # If audio does not exist, raise error
                if not os.path.exists(audio_path):
                    raise FileNotFoundError

                # Get folder path for new segmentation under asr_dir/pair-specific-dir/session-specific-dir/
                asr_session_dir = os.path.join(asr_dir,
                                               'pair' + str(pair_no),
                                               '_'.join(['pair' + str(pair_no),
                                                        lab,
                                                        session,
                                                        'fixed_segmentation'])
                                               )
                if not os.path.exists(asr_session_dir):
                    os.mkdir(asr_session_dir)

                # Get base filename for new segments
                segment_base = '_'.join(['pair' + str(pair_no),
                                         lab,
                                         session,
                                         'repaired_mono_noisered_segment'])

                # Store every relevant info in a tuple
                segmentation_list.append((srt_path,
                                          audio_path,
                                          asr_session_dir,
                                          segment_base))

    return segmentation_list


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


def segment_from_srt(srt_path, audio_path, output_dir, segment_basename):
    """

    :param srt_path:
    :param audio_path:
    :param output_dir:
    :param segment_basename:
    :return:
    """

    print(torchaudio.info(audio_path))
    waveform, sr = torchaudio.load(audio_path)

    # Check if audio is torch tensor, if not, try to cast it to tensor,
    # and hope that the error is informative if it fails
    if type(waveform).__name__ != 'Tensor':
        waveform = torch.tensor(waveform)
    # Check audio shape, should be (1, samples) or (samples, 1)
    if len(waveform.shape) != 2:
        raise ValueError('Arg long_audio should be 2D tensor!')
    if np.min(waveform.shape) != 1:
        raise ValueError('Arg long_audio should have shape (1, samples) or (samples, 1)!')
    # Normalize shape into (1, samples)
    if waveform.shape[1] == 1:
        waveform = waveform.T

    # Get timing info from srt
    start_times, end_times = srt_reader(srt_path)[0:2]

    # Transform subtitle time info to samples
    segment_starts = [round(t * sr) for t in start_times]
    segment_ends = [round(t * sr) for t in end_times]

    # Loop through segments, pass format-related args to torchaudio.save,
    # save out each segment
    all_segment_paths = []
    segment_idx = 0
    for start_idx, end_idx in zip(segment_starts, segment_ends):
        # Get segment from long_audio
        segment = waveform[:, start_idx: end_idx]
        # File name handling
        segment_filepath = os.path.join(output_dir, segment_basename) + str(segment_idx)
        all_segment_paths.append(segment_filepath)
        torchaudio.save(segment_filepath, segment,
                        sample_rate=sr,
                        channels_first=True,
                        format='wav',
                        encoding='PCM_S',
                        bits_per_sample=16)
        segment_idx = segment_idx + 1

    return all_segment_paths


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str, default='/media/gandalf/data_hdd/audio_transcription/data/', nargs='?',
                        help='Path to directory holding the audio segments (short wav files), '
                             'with subdirectories for each pair.')
    args = parser.parse_args()

    to_segment_list = info_for_segmentation(args.data_dir)
    for srt_p, audio_p, output_d, seg_base in to_segment_list:
        all_segment_paths = segment_from_srt(srt_p, audio_p, output_d, seg_base)


if __name__ == '__main__':
    main()
