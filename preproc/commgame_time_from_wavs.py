""""
Utility to query details about the length of each session for a given pair in the CommGame dataset

USAGE:
python commgame_times_from_wavs.py PAIRNO [PAIRNO2 PAIRNO3 ...] --audio_dir PATH_TO_WAVS

Timing data is read from the CommGame audio recordings. Input arg "--audio_dir" must point to the folder containing
the necessary wav files which follow the naming convention "pairPAIRNO_LAB_SESSION_repaired_mono.wav". The folder is
globbed recursively for the necessary files.

"""

from glob import glob
import argparse
import librosa
import os


def find_wavs(pair_no, audio_dir):

    pattern = os.path.join(audio_dir, '**/pair' + str(pair_no) + '_*_repaired_mono.wav')
    res = glob(pattern, recursive=True)
    return res


def get_audio_timing(wav_files):

    wav_length = []
    for wav in wav_files:
        y, sr = librosa.load(wav, sr=None)
        wav_length.append(len(y)/sr)
    return wav_length


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('pair_number', type=int, nargs=1,
                        help='Pair number, audio timing data is reported for this pair.')
    parser.add_argument('--audio_dir', type=str, default='/media/adamb/data_disk/CommGame/', nargs='?',
                        help='Path to noiseclip data csv file. Defaults to hardcoded path at '
                             '"/media/adamb/data_disk/CommGame/"')
    args = parser.parse_args()

    wav_files = find_wavs(args.pair_number[0], args.audio_dir)
    if not wav_files:
        raise FileNotFoundError('No wav file was found!')

    wav_length = get_audio_timing(wav_files)

    len_sum = sum(wav_length)

    print('\nFound ' + str(len(wav_files)) + ' audio files:')
    for w in wav_files:
        print(w)
    print('\nTotal time: ' + str(len_sum) + ' seconds (= '
          + str(len_sum/60) + ' mins = ' + str(len_sum/3600) + ' hours) \n')

    for z in zip(wav_files, wav_length):
        print(z)

    return


if __name__ == '__main__':
    main()
