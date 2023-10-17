"""

"""

from scipy.io import wavfile
import noisereduce as nr
import os
import argparse
import glob
from glob import glob
from noiseclip_generator import eat_noise_csv


def find_preprocessed_wav_files(pair, lab, data_dir):
    # glob recursively for preprocessed audio files in data_dir
    file_pattern = 'pair' + str(pair) + '_' + lab + '_*_repaired_mono.wav'
    wav_list = glob(os.path.join(data_dir, '**', file_pattern), recursive=True)

    return wav_list


def find_noise_wav_file(pair, lab, data_dir):
    # glob recursively for noise audio file in data_dir
    file_pattern = 'pair' + str(pair) + '_' + lab + '_noise.wav'
    noise_wav = glob(os.path.join(data_dir, '**', file_pattern), recursive=True)

    return noise_wav


def noise_red_params():
    # Define noise reduction parameters corresponding to different levels of background speech noise
    level_one = {
        'prop_decrease': 1.0,
        'n_std_thresh_stationary': 2.0,
        'stationary': True,
        'chunk_size': 16384,
        'padding': 8196,
        'n_fft': 1024,
        'n_jobs': 3
    }
    level_two = {
        'prop_decrease': 1.5,
        'n_std_thresh_stationary': 1.5,
        'stationary': True,
        'chunk_size': 16384,
        'padding': 8196,
        'n_fft': 1024,
        'n_jobs': 3
    }
    level_three = {
        'prop_decrease': 1.75,
        'n_std_thresh_stationary': 1.0,
        'stationary': True,
        'chunk_size': 16384,
        'padding': 8196,
        'n_fft': 1024,
        'n_jobs': 3
    }

    return [level_one, level_two, level_three]


def get_noise_reduced_audio(wav_file,
                            noise_file,
                            **kwargs):
    """
    Wrapper for calling the reduce_noise function from noisereduce on a wav file.
    Saves out the noise reduced audio as well, with name appended to mark noise reduction mode.
    """
    # load audio for noise reduction
    rate, data = wavfile.read(os.path.join(wav_file))
    # get noise data
    noise_rate, noise_data = wavfile.read(os.path.join(noise_file))
    # derive output file path
    path_parts = os.path.split(wav_file)
    file_parts = path_parts[1].split('.')
    new_wav_file = os.path.join(path_parts[0], file_parts[0] +
                                '_noisered.wav')
    # perform stationary noise reduction
    noise_reduced_audio = nr.reduce_noise(y=data.T,
                                          sr=rate,
                                          y_noise=noise_data.T,
                                          **kwargs)

    wavfile.write(new_wav_file, rate, noise_reduced_audio.T)

    return new_wav_file


def main():
    # Mandatory input argument defining pair numbers to to perform noise reduction for.
    # Optional argument (csv_path) for commgame noiseclip data .csv file,
    # that contains data helpful for the selection of noise reduction parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('pair_numbers', type=int, nargs='+',
                        help='Pair number(s), noise reduction is performed for these pairs. '
                             'If exactly two numbers are provided, a range of pair numbers is defined by treating '
                             'the two numbers as the first and last pair numbers.')
    parser.add_argument('--csv_path', type=str, default=None, nargs='?',
                        help='Path to noiseclip data csv file. The csv contains a "noise level" score for each pair '
                             'which can be used for noise reduction parameter selection.')
    args = parser.parse_args()

    # If there are two pair numbers, expand it into a range
    if len(args.pair_numbers) == 2:
        pair_numbers = list(range(args.pair_numbers[0], args.pair_numbers[1]+1))
    else:
        pair_numbers = args.pair_numbers

    # If no csv_path has been specified, it is set to None
    if not args.csv_path:
        csv_path = None
    else:
        csv_path = args.csv_path

    print('\nCalled noise_reduce_wrapper utility with args:')
    print('Pair numbers:', pair_numbers)
    print('csv file path:', csv_path)

    if not csv_path:
        raise UserWarning('As there was no csv file path provided, the same noise reduction ' 
                          'parameters will be applied to all files.')
    elif csv_path:
        print('Csv file will be parsed for pair-specific noise reduction parameters.')
        # Read and parse noiseclips csv
        csv_df = eat_noise_csv(csv_path)
        print('\nRead and parsed noiseclip data csv file.')

    # Load noise reduction parameter dictionaries, corresponding to different levels of noise reduction
    noise_params = noise_red_params()
    print('Loaded noise reduction params:')
    for i in range(len(noise_params)):
        print(noise_params[i])

    # Loop through pairs and labs, call noise_reduce function for each audio
    data_dir = '/media/adamb/data_disk/CommGame'
    lab_list = ['Mordor', 'Gondor']
    print('\nLooping through pair numbers:')
    for pair in pair_numbers:
        for lab_name in lab_list:

            print('\nWorking on pair ' + str(pair) + ', lab ' + lab_name)

            # Get audio files list and noise file path
            wav_files = find_preprocessed_wav_files(pair, lab_name, data_dir)
            noise_file = find_noise_wav_file(pair, lab_name, data_dir)
            if len(noise_file) != 1:
                raise FileExistsError('There are either multiple noise files associated with pair ' + str(pair)
                                      + ' lab ' + lab_name + ' or none!')
            else:
                noise_file = noise_file[0]
            print('Found ' + str(len(wav_files)) + ' wav files:')
            print(wav_files)
            print('Found corresponding noise file:')
            print(noise_file)

            # Select right noise reduction parameters, depending on the csv data if it was supplied
            if csv_path:
                csv_index_pattern = 'pair' + str(pair) + '_' + lab_name + '_freeConv_audio.wav'
                tmp_idx = csv_df['recording'] == csv_index_pattern
                noise_score = csv_df.loc[tmp_idx]['score'].to_numpy()[0]
            else:
                noise_score = 2
            noise_reduction_params = noise_params[noise_score-1]
            print('Noise reduction level for pair & lab:', noise_score)
            print('Noise reduction params:')
            print(noise_reduction_params)

            # Loop through each wav file, call noise reduction on it
            for wav in wav_files:
                print('Noise reduction on ', wav, '...')
                new_wav_file = get_noise_reduced_audio(wav, noise_file, **noise_reduction_params)
                print('Done, noise reduced audio saved out to', new_wav_file)


if __name__ == '__main__':
    main()
