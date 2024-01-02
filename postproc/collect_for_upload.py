"""
Small util for collecting BEAST2 output .srt files from the relatively complex files output folder / file structure.
Srt files are copied into a directory '/upload/pairPAIRNUMBER/' under --data_dir.

USAGE: python collect_for_upload PAIRNO_1 PAIRNO_2 ... PAIRNO_X --data_dir DATA_DIR

If exactly two pair numbers are provided, they are expanded into a range (inclusive on both ends).
"""

import os
import argparse
from glob import glob
from shutil import copy


def main():
    # Arguments for pair numbers, and audio / asr directory
    parser = argparse.ArgumentParser()
    parser.add_argument('pair_numbers', type=int, nargs='+',
                        help='Pair numbers, determines which audio (wav) files are selected for analysis.'
                             'If exactly two numbers are provided, a range of pair numbers is defined by treating '
                             'the two numbers as the first and last pair numbers.')
    parser.add_argument('--data_dir', type=str, default=os.getcwd(), nargs='?',
                        help='Path to directory holding the data, '
                             'with subdirectories "/raw", "/noise_reduced", "/asr", and "/upload".')
    args = parser.parse_args()

    # Checks for necessary directories
    if not os.path.exists(args.data_dir):
        raise NotADirectoryError('Input arg --data_dir is not a valid path!')
    if (not os.path.exists(os.path.join(args.data_dir, 'raw'))
            or not os.path.exists(os.path.join(args.data_dir, 'noise_reduced'))
            or not os.path.exists(os.path.join(args.data_dir, 'upload'))
            or not os.path.exists(os.path.join(args.data_dir, 'asr'))):
        raise NotADirectoryError('Input arg --data_dir does not have subdirectories "/raw", '
                                 '"/noise_reduced", upload, or "/asr"!')

    # If there are two pair numbers, expand it into a range
    if len(args.pair_numbers) == 2:
        pair_numbers = list(range(args.pair_numbers[0], args.pair_numbers[1]+1))
    else:
        pair_numbers = args.pair_numbers
    data_dir = args.data_dir

    print('\nCalled collect_for_upload with args:')
    print('Pair numbers: ' + str(pair_numbers))
    print('Data directory: ' + data_dir)

    for pair in pair_numbers:
        print('\nWorking on pair ' + str(pair))
        # Create directory for collection if it does not exist yet
        target_dir = os.path.join(data_dir, 'upload', 'pair' + str(pair))
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
            print('Created collection directory at ', target_dir)
        # Find all srt files
        asr_dir = os.path.join(data_dir, 'asr', 'pair' + str(pair))
        srt_files = glob(asr_dir + '/*/*beast2.srt')
        if not srt_files:
            raise FileNotFoundError('Cannot find any .srt files for pair ' + str(pair))
        else:
            print('Found ' + str(len(srt_files)) + ' transcript files:')
            for s in srt_files:
                print(s)
        # Copy all srt files
        print('Copying srt files to ' + target_dir)
        for srt in srt_files:
            basename = os.path.basename(srt)
            target_path = os.path.join(target_dir, basename)
            copy(srt, target_path)
        print('Done.')


if __name__ == '__main__':
    main()
