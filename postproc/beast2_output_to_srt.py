"""
Utility to transform BEAST2 ASR output to a subtitles (srt) file.

USAGE: python beast2_output_to_srt.py --target_dir PATH

DEPENDENCIES:
Numpy and srt

NOTES:
- target_dir is searched recursively for any file matching the standard BEAST2 output naming convention
used in the CommGame data analysis pipeline: "pair*_beast2_results.txt"
- For each BEAST2 output file, the corresponding segmentation file is used for srt timings. These follow the
naming convention pairPAIRNO_LAB_SESSION_repaired_mono_noisered_segmentation_samples.npz

!!!! Everything is very sensitive to errors in file names !!!!

"""

import os
import re
import srt
from glob import glob
from datetime import timedelta
import numpy as np
import argparse


def find_files(results_dir, results_pattern='**/pair*_beast2_results.txt'):
    """
    Find all files fitting the pattern recursively, with glob.
    For each result, find also the corresponding "*_segmentation_samples.npz" file as well.
    :param results_dir:     Str, path to folder to search recursively.
    :param results_pattern: Str, pattern for glob search.
    :return: result_files:  List of strings, each element a path to a file matching the pattern.
    :return: seg_files:     List of strings, each element a path to a "*_segmentation_samples.npz" file.
    """
    # Find BEAST2 output text files
    result_files = glob(os.path.join(results_dir, results_pattern), recursive=True)
    # Find corresponding segmentation files (numpy .npz files)
    seg_files = []
    for file in result_files:
        seg = file[0:-18] + 'segmentation_samples.npz'  # very, very specific !!!
        if not os.path.exists(seg):
            raise FileNotFoundError('Cannot find segmentation file for ' + file + ' at ' + seg + '!')
        else:
            seg_files.append(seg)

    return result_files, seg_files


def extract_lines(results_path):
    """
    Function to process a WER stats style BEAST2 output text file. Extracts and cleans the transcripts and sorts
    them according to the speech segment numbers.

    :param results_path:         String, path to BEAST2 output (WER stats formatted) text file.
    :return: segment_no:         List of integers, contains the speech segment numbers for corresponding elements in
                                 transcripts_sorted
    :return: transcripts_sorted: List of strings, with each string containing the transcript of a speech segment.
    """

    # Read file
    with open(results_path, 'r') as file_in:
        lines = file_in.readlines()

    # Lists that hold the transcripts and the corresponding segment numbers, respectively
    transcripts = []
    segment_no = []

    # Prepare regexes for matching lines that specify segment number
    regex_start = re.compile('^pair.*_segment')
    regex_segment_no = re.compile('_segment[0-9]+,')

    # Loop through lines, mark the ones with a valid segment file name
    for string_idx, string in enumerate(lines):
        line_match = regex_start.match(string)
        if line_match:
            # Get first which segment it belongs to
            segment_no_match = regex_segment_no.search(string)
            segment_no_string = string[segment_no_match.span()[0]+8: segment_no_match.span()[1]-1]  # heavily specific!!
            # Get corresponding transcript
            transcript_string = lines[string_idx + 3]  # again, heavily specific
            # Append to result lists
            segment_no.append(segment_no_string)
            transcripts.append(transcript_string)

    # Clean the transcription lines
    regex_cleaning = '[^A-Za-z0-9 éáöőüűúíó]+'
    transcripts_clean = []
    for string in transcripts:
        string_clean = re.sub(regex_cleaning, '', string)
        string_clean = ' '.join(string_clean.split())
        transcripts_clean.append(string_clean)

    # Sort!
    segment_no = [int(s) for s in segment_no]
    transcripts_sorted = [t for s, t in sorted(zip(segment_no, transcripts_clean), key=lambda pair: pair[0])]
    segment_no.sort()

    return segment_no, transcripts_sorted


def compose_srt_from_lists(subtitle_indices, subtitles_list, start_times_list, end_times_list):
    """
    Function to generate an SRT-formatted string from transcriptions, their start and end times, and their indices.
    !!! Elements of input lists must correspond to the same subtitles !!!!
    :param subtitle_indices:  List of integers, subtitle indices (temporal order).
    :param subtitles_list:    List of strings, each string a subtitle.
    :param start_times_list:  List of timedelta objects, each marking the start time of the corresponding subtitle.
    :param end_times_list:    List of timedelta objects, each marking the end time of the corresponding subtitle.
    :return: srt_string:      String, SRT-formatted, composed of all subtitles.
    """
    # Sanity checks
    if (len(subtitles_list) != len(subtitle_indices)
            or len(subtitle_indices) != len(start_times_list)
            or len(subtitle_indices) != len(end_times_list)):
        raise ValueError('Input args should be lists of equal length!')
    # Generate subtitle objects
    subtitles = []
    for seq_no, start, end, text in zip(subtitle_indices, start_times_list, end_times_list, subtitles_list):
        subtitles.append(srt.Subtitle(index=seq_no,
                                      start=timedelta(seconds=start),
                                      end=timedelta(seconds=end),
                                      content=text))
    # Compose
    srt_string = srt.compose(subtitles)

    return srt_string


def load_segmentation_data(seg_file, sampling_rate=16000):
    """
    Function to load and transform the audio segmentation data from a "*_segmentation_samples.npz" file.
    :param seg_file:        Str, path to file.
    :param sampling_rate:   Int, sampling rate in Hz.
    :return:segment_starts: List of timedelta objects, each one marking the start time of an audio segment.
    :return:segment_ends:   List of timedelta objects, each one marking the end time of an audio segment.
    """
    # Load numpy arrays
    tmp = np.load(seg_file)
    segment_starts = tmp['segment_starts']
    segment_ends = tmp['segment_ends']
    # Transform to list of seconds
    segment_starts = np.asarray(segment_starts/sampling_rate).tolist()
    segment_ends = np.asarray(segment_ends/sampling_rate).tolist()

    return segment_starts, segment_ends


def main():
    # Input argument handling
    # One mandatory and one optional argument, defining pair number(s) and the path to the audio (wav) files.
    # Arg args.pair_number is a list with one or more elements while args.audio_dir is string (path to dir).
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, default=os.getcwd(), nargs='?',
                        help='Path to folder which should be searched recursively for BEAST2 output files.')
    args = parser.parse_args()

    # Sanity checks
    if not os.path.exists(args.target_dir):
        raise ValueError('Input arg --target_dir is not a valid path!')

    print('\nCalled beast2_output_to_srt.py with target dir:')
    print(args.target_dir)

    # Get lists of files to work with
    transcription_files, seg_files = find_files(args.target_dir,
                                                results_pattern='**/pair*_beast2_results.txt')
    print('\nFound ' + str(len(transcription_files)) + ' BEAST2 output files:')
    print(transcription_files)
    print('\nFound ' + str(len(seg_files)) + ' segmentation files:')
    print(seg_files)

    # Loop through outputs
    for transcript, segmentation in zip(transcription_files, seg_files):
        print('\nWorking on files:')
        print(transcript, segmentation)

        # Parse transcript file
        segment_no, transcripts_sorted = extract_lines(transcript)

        # Parse segmentation file
        segment_starts, segment_ends = load_segmentation_data(segmentation, sampling_rate=16000)

        # Compose srt
        srt_string = compose_srt_from_lists(segment_no,
                                            transcripts_sorted,
                                            segment_starts,
                                            segment_ends)

        # Generate SRT file path and write out srt
        srt_path = transcript[0:-19] + '_beast2.srt'
        with open(srt_path, 'w') as file_out:
            file_out.write(srt_string)

        print('Done, subtitles written to srt at')
        print(srt_path)


if __name__ == '__main__':
    main()
