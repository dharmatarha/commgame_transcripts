"""
Utility / module for transforming pairs of srt files to a simple text format after applying simple cleaning steps.

USAGE:
python3 srt_to_text.py DATA_DIR PAIRNO [PAIRNO ... [PAIRNO]]

where DATA_DIR holds pair-specific subfolders named "pairPAIRNO" which in turn contain pair-specific srt files, and
PAIRNO is a pair number. Arbitrary number of pair numbers might be provided to the script. If exactly two pair numbers
are provided, they are treated as the start and the end of a range (inclusive on both ends).

Outputs are .txt files in the same folders that hold the srt files.

By default subtitles from pairs of srt files are:
(1) Sorted according to their start time;
(2) Cleaned from extra whitespaces and from common errors / misspellings defined in REPLACEMENTS constant;
(3) Cleaned from tags of non-linguistic vocal behavior (laughter, hesitation, hums).

There are simple functions included to filter out potential backchannels and very short speech turns, but these are
not applied by default.

Notes:
- Include options to call additional cleaning steps, with further options specifying main parameters.
- Look into spaCy methods we could apply as well after an import.

"""


import argparse
import os
import srt
import copy
import re
from glob import glob


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
    '<laugh<': '<laugh>',
    '>laugh>': '<laugh>',
    '>laugh<': '<laugh>',
    '<hes<': '<hes>',
    '>hes>': '<hes>',
    '>hes<': '<hes>',
    '<hum<': '<hum>',
    '>hum>': '<hum>',
    '>hum<': '<hum>'}


def srt_to_txt(srt_files, speaker_tags=None):
    """
    Function to generate an sorted list of subtitles from a list of srt files. Srt files might have temporally
    overlapping content. Output list is sorted according to the start time of each subtitle.

    :param srt_files:     List of srt file paths.
    :param speaker_tags:  Iterable with one string for each srt file, e.g. ('SPEAKER1: ', 'SPEAKER2: '). The tags are
                          inserted at the beginning of each subtitle (e.g. if subtitle content was "Hello!", it becomes
                          "SPEAKER1: Hello!"). Defaults to the tuple ('Speaker1: ', 'Speaker2: ', ... , 'SpeakerN: ')
                          where N = len(srt_files).
    :return: all_subs:    List of all subtitles parsed from the srt files in input arg "srt_files". Sorted according to
                          subtitle start times.
    """
    # Check inputs.
    if not speaker_tags:
        speaker_tags = tuple(['Speaker' + str(i+1) + ': ' for i in range(len(srt_files))])
    if len(srt_files) != len(speaker_tags):
        raise ValueError('Input args srt_files and speaker_tags must have equal length')
    # Open and parse srt files. Var "subs" is a list of lists of subtitle objects.
    srt_subs_list = []
    for srt_f in srt_files:
        with open(srt_f, 'r') as f_in:
            srt_subs_list.append(list(srt.parse(f_in)))
    # Add speaker tags to subtitle strings.
    for srt_idx, srt_subs in enumerate(srt_subs_list):
        for sub in srt_subs:
            sub.content = ' '.join([speaker_tags[srt_idx], sub.content])
    # Merge all subtitles into one sorted list. Sorting is according to start time.
    all_subs = [subtitle for srt_subs in srt_subs_list for subtitle in srt_subs]
    all_subs.sort(key=lambda x: x.start)

    return all_subs


def replace_patterns_in_subs(subtitles_list, replacements=REPLACEMENTS):
    """
    Helper function to traverse a list of subtitle objects and perform the set of string replacements defined in
    "replacements". Returns a list of subtitle objects. Replacements are done with simple string.replace(), no fancy regex patterns to
    use here.
    Works on deep-copied version of input list to avoid unintentional replacement-in-place.
    :param subtitles_list:  List of srt subtitle objects.
    :param replacements:    Dictionary, with keys and values corresponding to strings to replace and their
                            replacements, respectively. Defaults to module-level constant REPLACEMENTS.
    :return:                List of srt subtitle objects.
    """
    subs_list = copy.deepcopy(subtitles_list)  # Avoid messing up input arg list in place
    for sub in subs_list:
        for r_key in replacements:
            sub.content.replace(r_key, replacements[r_key])
    return subs_list


def filter_short_subs(subtitles_list, min_length_s=1):
    """
    Helper function to delete subtitles from a list of subtitles if they are shorter than some threshold.
    Works on deep-copied version of input list to avoid unintentional deletion-in-place.
    :param subtitles_list:  List of srt subtitle objects.
    :param min_length_s:    Minimum length in seconds. Defaults to one second.
    :return:                List of srt subtitle objects.
    """
    subs_list = copy.deepcopy(subtitles_list)  # Avoid messing up input arg list in place
    subs_list = [sub for sub in subs_list if (sub.end-sub.start).total_seconds() >= min_length_s]
    return subs_list


def filter_backchannel_subs(subtitles_list, max_length_s=1.5, padding_s=1):
    """
    Helper function to delete short, "backchannel" type subtitles from a list of srt subtitle objects.
    Backchannel is defined as short speech uttered during a longer, "dominant" speech segment by some other speaker.
    Input arg "subtitles_list" MUST BE SORTED by subtitle start times, otherwise the current function does not work!
    Works on deep-copied version of input list to avoid unintentional deletion-in-place.
    :param subtitles_list:  List of srt subtitle objects.
    :param max_length_s:    Maximum length of a speech turn for labeling it as backchannel, in seconds. Defaults to 1.5.
    :param padding_s:       Margin in seconds for determining if a speech turn is indeed during another, longer
                            speech turn. A turn is only labelled backchannel, if its start is at least "padding_s"
                            after the start of the previous speech turn, and its end is at least "padding_s" before
                            the end of the previous speech turn. Defaults to 1.
    """
    subs_list = copy.deepcopy(subtitles_list)  # Avoid messing up input arg list in place
    # Sanity check, is the subtitles list sorted?
    sorted_bool = all([subtitles_list[i+1].start >= subtitles_list[i].start for i in range(len(subtitles_list)-1)])
    if not sorted_bool:
        raise AssertionError('Input arg "subtitles_list" MUST be sorted according to subtitle start times!')
    # Loop through subtitles, always compare current with next one
    for sub_idx, sub in enumerate(subs_list):
        # If we are not at the end of the subtitle list, compare the timing of the current and the next subtitle.
        if sub_idx != len(subs_list)-1:
            sub_next = subs_list[sub_idx + 1]
            # If the next subtitle happens during the time of the current subtitle considering padding as well),
            # and it is considered short enough, delete it.
            if (sub_next.end.total_seconds() + padding_s <= sub.end.total_seconds()) and \
                    (sub_next.start.total_seconds()-padding_s >= sub.start.total_seconds()) and \
                    (sub_next.end-sub_next.start).total_seconds() <= max_length_s:
                subs_list.pop(sub_idx + 1)
    return subs_list


def filter_tags_in_subs(subtitles_list, tags=('<laugh>', '<hes>', '<hum>')):
    """
    Helper function to delete tags from list of srt subtitle objects. Tags to delete are specified in "tags".
    If the only content in a subtitle is a tag, the whole object is deleted.
    Works on deep-copied version of input list to avoid unintentional deletion-in-place.
    :param subtitles_list:  List of srt subtitle objects.
    :param tags:            Iterable of strings. Defaults to ('<laugh>', '<hes>', '<hum>').
    :return:                List of srt subtitle objects.
    """
    subs_list = copy.deepcopy(subtitles_list)  # Avoid messing up input arg list in place
    for sub in subs_list:
        for t in tags:
            sub.content = sub.content.replace(t, '')
    subs_list = [s for s in subs_list if s.content]
    return subs_list


def norm_whitespaces_in_subs(subtitles_list):
    """
    Helper function cleaning up leading, trailing and sequential whitespaces in subtitle content strings.
    Works on deep-copied version of input list to avoid unintentional tampering-in-place.
    :param subtitles_list:  List of srt subtitle objects.
    :return: subs_list:     List of srt subtitle objects.
    """
    subs_list = copy.deepcopy(subtitles_list)  # Avoid messing up input arg list in place
    too_many_whitespaces_regex = re.compile('\s{2,}')
    for sub in subs_list:
        sub.content = sub.content.strip()
        sub.content = too_many_whitespaces_regex.sub(' ', sub.content)
    return subs_list


def find_srt_pairs(srt_dir, pair_no, sessions=('BG1', 'BG2', 'BG3', 'BG4', 'BG5', 'BG6', 'freeConv')):
    """
    Helper function to find pairs of corresponding srt subtitle files for given "pair_no" and "sessions" in "srt_dir".
    Assumes srt filenames following the convention:
    pair[PAIRNO]_[LAB]_[SESSION]_repaired_mono_noisered*.srt
    where PAIRNO is the pair number, LAB is one of ('Mordor', 'Gondor'), and SESSION is one of
    ('BG1', 'BG2', 'BG3', 'BG4', 'BG5', 'BG6', 'freeConv').

    :param srt_dir:         String, path to folder holding the srt files.
    :param pair_no:         Numeric, pair number for finding srt files.
    :param sessions:        Iterable of strings, session names for finding srt files. Defaults to
                            ('BG1', 'BG2', 'BG3', 'BG4', 'BG5', 'BG6', 'freeConv')
    :return: srt_files:     List of tuples, with each tuple containing Mordor and Gondor srt files corresponding to
                            the same pair and session. The ordering is always ('Mordor srt part', 'Gondor srt path').
    :return: sessions_list: List of strings, contains the session names corresponding to srt file pairs.
    """
    srt_files = []
    sessions_list = []
    # Loop through the requested sessions.
    for ses in sessions:
        # Find Mordor lab srt file.
        srt_mordor_pattern = os.path.join(srt_dir,
                                          '_'.join(['pair' + str(pair_no),
                                                    'Mordor', ses,
                                                    'repaired_mono_noisered*.srt'])
                                          )
        srt_mordor_path = glob(srt_mordor_pattern)
        if not srt_mordor_path:
            raise FileNotFoundError('Cannot find Mordor srt file for ' + ses + '!')
        elif len(srt_mordor_path) > 1:
            raise FileNotFoundError('Too many Mordor srt files for ' + ses + '!')
        else:
            srt_mordor_path = srt_mordor_path[0]
        # Find Gondor lab srt file.
        srt_gondor_pattern = os.path.join(srt_dir,
                                          '_'.join(['pair' + str(pair_no),
                                                    'Gondor', ses,
                                                    'repaired_mono_noisered*.srt'])
                                          )
        srt_gondor_path = glob(srt_gondor_pattern)
        if not srt_gondor_path:
            raise FileNotFoundError('Cannot find Gondor srt file for ' + ses + '!')
        elif len(srt_gondor_path) > 1:
            raise FileNotFoundError('Too many Gondor srt files for ' + ses + '!')
        else:
            srt_gondor_path = srt_gondor_path[0]
        # Append the two files as a tuple to output list.
        srt_files.append((srt_mordor_path, srt_gondor_path))
        # Append the session name to output list.
        sessions_list.append(ses)

    return srt_files, sessions_list


def main():
    # Input argument handling. First positional argument is for the folder with pair-level subfolders holding the
    # srt files. Second is for pair numbers and can eat an arbitrary number of them. If exactly two pair numbers are
    # supplied, they are treated as start and end of a range (inclusive on both ends).
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='Path to main data folder. "data_dir" is expected to contain pair-specific subfolders '
                        '(e.g. "data_dir/pair999") which themselves hold the pair-specific srt files.')
    parser.add_argument('pair_numbers', type=int, nargs='+',
                        help='Pair numbers, determines which srt files are selected.'
                             'If exactly two numbers are provided, a range of pair numbers is defined by treating '
                             'the two numbers as the first and last pair numbers.')
    args = parser.parse_args()

    # Sanity check on "data_dir".
    if not os.path.exists(args.data_dir):
        raise NotADirectoryError('Input arg --audio_dir is not a valid path!')
    # If there are two pair numbers, expand them into a range.
    if len(args.pair_numbers) == 2:
        pair_numbers = list(range(args.pair_numbers[0], args.pair_numbers[1]+1))
    else:
        pair_numbers = args.pair_numbers

    print('\nCalled srt_to_test with args:')
    print('Main data folder: ' + args.data_dir)
    print('Pair numbers: ' + str(pair_numbers))

    # Loop through all requested pairs.
    for pair in pair_numbers:

        # Find srt files for given pair.
        pair_dir = os.path.join(args.data_dir, 'pair' + str(pair))
        print('\nListing srt files for pair ' + str(pair) + ' in ' + pair_dir + '.')
        pair_srt_files, sessions_list = find_srt_pairs(pair_dir, pair)
        print('Found srt files for ' + str(len(pair_srt_files)) + ' sessions:')
        print(pair_srt_files)

        # Loop through pairs of srt files.
        for ses_idx, session_files in enumerate(pair_srt_files):
            # Read in srt files and clean up the subtitles:
            # (1) replace common misspellings; (2) remove tags; (3) clear extra whitespaces.
            print('Reading srt files, cleaning up subtitles...')
            subtitles_list = srt_to_txt(session_files)
            subtitles_list_repl = replace_patterns_in_subs(subtitles_list)
            subtitles_list_notags = filter_tags_in_subs(subtitles_list_repl)
            subtitles_list_final = norm_whitespaces_in_subs(subtitles_list_notags)

            # Write out final list of subtitles as txt.
            subs_list = [sub.content + '\n' for sub in subtitles_list_final]
            output_filepath = os.path.join(pair_dir,
                                           '_'.join(['pair' + str(pair),
                                                     sessions_list[ses_idx],
                                                     'combined_srt_files.txt'
                                                     ])
                                           )
            with open(output_filepath, 'w') as f_out:
                f_out.writelines(subs_list)
            print('Wrote out subtitles to text file at', output_filepath)


if __name__ == '__main__':
    main()
