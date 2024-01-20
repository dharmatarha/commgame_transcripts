"""
Utility for generating noise clips from freeConv raw audios in the CommGame project.

USAGE:
python3 noiseclip_generator.py [PAIRNUMBERS] --csv_path [CSVFILEPATH]

If two PAIRNUMBERS are provided, a they are treated as the first and last pair numbers in a range.

REQUIRES:
- A commgame noiseclip .csv file containing start and end times for noise clips for each raw audio.
Usually this is called commgame_noiseclip.csv.
- Raw audio files accessible somewhere in the hardcoded data_dir folder, following the CommGame naming convention
"pair[PAIRNUMBER]_[LAB]_freeConv_audio.wav"

The program saves out a noise audio clip for each pair number - lab combo, named "pair[PAIRNUMBER]_[LAB]_noise.wav",
saved out into the folder containing the corresponding raw audio.

NOTES:
- Data folder path is hardcoded in main()
- Lab names are hardcoded in main()
- Default (fallback) csv_path is hardcoded in main()
"""

import argparse
from scipy.io import wavfile
import os
import glob
import pandas as pd
import numpy as np
from glob import glob


def eat_noise_csv(csv_path):
    """
    Read in and parse the .csv file holding noiseclip data for CommGame freeConv raw audios.
    The csv must have four columns corresponding to (1) Audio file name (recording);
    (2) Partner speech noise level (score); (3) Noise audioclip start in format "minutes:seconds";
    (4) Noise audioclip end in format "minutes:seconds"

    Noiseclips should be for raw freeConv audio files with naming convention "pair[PAIRNUMBER]_[LAB]_freeConv_audio.wav"

    :param csv_path:  String, path to .csv file.
    :return: csv_df:  Pandas dataframe holding all information from .csv file. Noiseclip start and end times
                      are transformed into seconds (integers).
    """
    csv_df = pd.read_csv(csv_path, sep=',', header=0, names=['recording', 'score', 'noise_start', 'noise_end'])

    # Turn columns "noise_end" and "noise_start" into numeric ones, with seconds instead of string "minutes:seconds"
    noise_end = csv_df.noise_end[:]
    noise_end_num = np.zeros((len(noise_end), 1))
    for idx, n_e in enumerate(noise_end):
        tmp_list = [int(i) for i in n_e.split(':')]
        noise_end_num[idx] = tmp_list[0]*60 + tmp_list[1] - 1
    csv_df.noise_end = noise_end_num.squeeze(1).astype('int64')
    # noise start
    noise_start = csv_df.noise_start[:]
    noise_start_num = np.zeros((len(noise_start), 1))
    for idx, n_s in enumerate(noise_start):
        tmp_list = [int(i) for i in n_s.split(':')]
        noise_start_num[idx] = tmp_list[0]*60 + tmp_list[1] + 1
    csv_df.noise_start = noise_start_num.squeeze(1).astype('int64')

    return csv_df


def generate_noiseclip(pair_number, lab_name, csv_df, data_dir='/media/adamb/data_disk/CommGame/'):
    """
    Function to generate a noise clip (audio containing speech signal from partner) wav file for a given pair and lab.
    Depends on CommGame naming conventions: raw audio is named "pair[PAIRNUMBER]_[LAB]_freeConv_audio.wav".
    Generates a .wav file named "pair[PAIRNUMBER]_[LAB]_noise.wav". Noise .wav file is saved to the same folder as
    the original raw audio.

    :param pair_number: Integer, pair number.
    :param lab_name:    String, lab name, either 'Mordor' or 'Gondor'.
    :param csv_df:      Pandas dataframe, output of eat_noise_csv()
    :param data_dir:    String, path to the folder to be searched recursively for the necessary raw audio .wav file.
    :return: noise_wav_path: String, path to the noise .wav file saved out.
    """
    # Find relevant audio file for noise clip
#    pair_audio_pattern = '**/pair' + str(pair_number) + '_' + lab_name + '_freeConv_repaired_mono.wav'  # to be used with *_repaired_mono.wav files
    pair_audio_pattern = '**/pair' + str(pair_number) + '_' + lab_name + '_freeConv_audio.wav'  # to be used with *_audio.wav files
    audio_file = glob(os.path.join(data_dir, pair_audio_pattern), recursive=True)
    if not audio_file:
        raise FileNotFoundError('Could not find audio file for pattern ' + pair_audio_pattern)
    else:
        audio_file = audio_file[0]
        print('Found audio file at ' + audio_file)
    # Find noise clip start and end times
    row_mask = csv_df['recording'] == os.path.basename(audio_file)
    noise_start = csv_df.noise_start[row_mask].to_numpy()[0]
    noise_end = csv_df.noise_end[row_mask].to_numpy()[0]
    # Cut out noise clip, transform to mono
    rate, data = wavfile.read(audio_file)
    noiseclip = data[noise_start * rate + 1: noise_end * rate, :]  # to be used with *_audio.wav files
#    noiseclip = data[noise_start*rate+1: noise_end*rate]  # to be used with *_repaired_mono.wav files
    noiseclip_mono = np.mean(noiseclip, axis=1)              # to be used with *_audio.wav files
    noiseclip_mono = np.expand_dims(noiseclip_mono, axis=1)  # to be used with *_audio.wav files
#    noiseclip_mono = np.expand_dims(noiseclip, axis=1)    # to be used with *_repaired_mono.wav files
    # Save out noise clip
#    noise_wav_path = os.path.join(os.path.split(audio_file)[0], os.path.basename(audio_file)[0: -26] + 'noise.wav')  # to be used with *_repaired_mono.wav files
    noise_wav_path = os.path.join(os.path.split(audio_file)[0], os.path.basename(audio_file)[0: -18] + 'noise.wav')  # to be used with *_audio.wav files
    noiseclip_mono = noiseclip_mono.astype('int16')  # needed for PCM 16bit wav format
    wavfile.write(noise_wav_path, rate, noiseclip_mono)

    return noise_wav_path


def main():
    # Mandatory input argument defining pair numbers to generate noise clips for.
    # Optional argument (csv_path) for commgame noiseclip data .csv file.
    parser = argparse.ArgumentParser()
    parser.add_argument('pair_numbers', type=int, nargs='+',
                        help='Pair number(s), noise clips are generated for these pairs. '
                             'If exactly two numbers are provided, a range of pair numbers is defined by treating '
                             'the two numbers as the first and last pair numbers.')
    parser.add_argument('--csv_path', type=str, default=None, nargs='?',
                        help='Path to noiseclip data csv file. Defaults to hardcoded path.')
    args = parser.parse_args()

    # If there are two pair numbers, expand it into a range
    if len(args.pair_numbers) == 2:
        pair_numbers = list(range(args.pair_numbers[0], args.pair_numbers[1]+1))
    else:
        pair_numbers = args.pair_numbers

    # If no csv_path has been specified, fallback to a default path
    if not args.csv_path:
        csv_path = '/media/adamb/data_disk/CommGame/commgame_noiseclips.csv'
    else:
        csv_path = args.csv_path

    print('\nCalled noiseclip_generator utility with args:')
    print('Pair numbers:', pair_numbers)
    print('csv file path:', csv_path)

    # Read and parse noiseclips csv
    csv_df = eat_noise_csv(csv_path)
    print('\nRead and parsed noiseclip data csv file.')

    # Loop through pairs and labs, call noiseclip generator function for each audio
    data_dir = '/media/adamb/data_disk/CommGame'
    lab_list = ['Mordor', 'Gondor']
    print('\nLooping through pair numbers:')
    for pair in pair_numbers:
        for lab_name in lab_list:
            print('Working on pair ' + str(pair) + ', lab ' + lab_name)
            noise_wav_path = generate_noiseclip(pair, lab_name, csv_df, data_dir)
            print('Generated noise wav:')
            print(noise_wav_path)

    print('\n\nFinished with all pairs!')
    print('So long, and thanks for all the fish!')


if __name__ == '__main__':
    main()
