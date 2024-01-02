# Transcription generation pipeline for CommGame experiment audio data (Hungarian)

A pipeline for automated transcription of somewhat noisy long audio recordings. Our use case is a series of cognitive / neuroscience experiments focusing on human verbal interactions (CommGame project).  

Participants in the CommGame experiments have been recorded with HSE-150/SK stage microphones attached to IMG Stageline EMA-1 adapters, sampled by ESI Maya22 audio interfaces at 44100 Hz. Audio is stored in uncompressed 16-bit PCM format.  

For transcript generation, audio is first (1) preprocessed, then (2) filtered for background noise, optionally (3) further filtered in terms of overall power, then (4) segmented by a voice detector, and finally (5) transcribed by an ASR model.  
The pipeline relies on the following great projects:  

* [noisereduce package](https://github.com/timsainb/noisereduce)
* [Silero Voice Activity Detector](https://github.com/snakers4/silero-vad)
* [The BEA Speech Transcriber (BEAST2) model](https://phon.nytud.hu/bea/bea-base.html?lang=en)
  
The pipeline is prepared for Ubuntu 22.04 LTS as the target system. Matlab part has been tested with 2017a and above, python parts have been tested with 3.10. In principle, Matlab functions should be easy to adapt to Octave but this has never been tested.


### (1) Preprocessing raw audio  
**Matlab / Octave functions.**  
- Audio is checked for missing segments (buffer underflow events during streaming) and deviations from nominal sampling rate (which is a strangely common problem), with resampling where necessary. This procedure reconstructs the true timeline of the recorded audio as best as we can do it offline. Implemented in `audioRepair.m`, wrapped in `audioRepairWrapper.m` for batch processing. Sample call with our default parameters:  
      ```audioRepair('/media/adamb/data_disk/CommGame/pair99', 99, 'freeConv', 0.020, 225, 0.5, 44100)```
- The outputs are two mono wav files containing the preprocessed audio streams from the two labs. They are trimmed to start at the `sharedStartTime` timestamp and to end when the shorter of the two finishes. Standard output naming follows this convention:  
```pairPAIRNUMBER_LABNAME_SESSION_repaired_mono.wav```  
(e.g. ```pair99_Mordor_freeConv_repaired_mono.wav```)

### (2) Noise reduction  
**Python using [noisereduce](https://github.com/timsainb/noisereduce).**  

   Recordings were often sensitive enough to pick up not only speech from the participant wearing the microphone but also the other participant's speech streamed from the other lab and played from a speaker (crosstalk).  
   The noise reduction step aims to eliminate both crosstalk and the occasional line noise. Could be an optional step depending on the amount of crosstalk but has been used for all CommGame audio analysis so far.  
- A csv file (`commgame_noiseclips.csv`) contains the start and end times of a noise (crosstalk) segment within each freeConv recording. First a noise clip (wav) is cut from each freeConv, that is, one for each participant using `noiseclip_generator.py`. Sample call for pair 99:  
```python noiseclip_generator.py 99 --csv_path CSV_PATH```
- The generated noise clips are used for noise reduction across all audio files from the same speaker (all sessions). This part is handled by `noise_reduce_wrapper.py` which calls the `reduce_noise` method from the `nosiereduce` package repeatedly, for all audio files. We use the stationary method, with the parameters coded into `noise_red_params` within `noise_reduce_wrapper.py`. There are slightly different parameters defined for different levels of noise reduction. The csv file `commgame_noiseclips.csv` contains a rating regarding the degree of crosstalk for the recordings from each participant, `noise_reduce_wrapper` relies on this information for setting noise reduction parameters.
Sample call for `noise_reduce_wrapper.py` for pairs 90 to 99:  
```python noise_reduce_wrapper.py 90 99 --csv_path CSV_PATH```  
 - Standard output naming after noise reduction follows this convention:  
```pairPAIRNUMBER_LABNAME_SESSION_repaired_mono_noisered.wav```  
(e.g. ```pair99_Mordor_freeConv_repaired_mono_noisered.wav```)

### (3) RMS-based filtering
**Python (librosa, soundfile, matplotlib, numpy)**  
Optional step, used whenever the output from noise reduction is deemed still too noisy for transcription. Very specific to our use case where crosstalk is the biggest problem in terms of transcription, and where crosstalk is usually much softer than speech from the primary speaker.  
The goal of this step is to differentiate the crosstalk from target speech based on power (RMS) and further reduce power for segments probably belonging to crosstalk towards a noise floor.  
- **Overlook**:
  - Per-frame RMS is estimated via STFT for all audio signals from the same participant (default frame length is 2048 samples, with 50 % overlap).
  - Log RMS values are depicted in a histogram and the user is prompted to provide two threshold values (a higher and a lower). Usually it is easy to spot the differences between primary speech signal, crosstalk, and line noise on the histogram. The higher threshold should correspond to a ~10% cumulative cutoff of the primary speech signal distribution (left tail), while the lower threshold should mark a ~60% cumulative cutoff for the crosstalk distribution (slightly right from the center). These are derived from practice and can vary form speaker-to-speaker for optimal results. Note that we rely on visual inspection because automatic detection (fitting) of a Gaussian mixture had poor reliability in our data.
  - The RMS time-series is windowed (default window length is 11 frames, with 50 % (rounded down) overlap), and each window is characterized by its mean RMS.
  - Mean RMS values between the lower and higher cutoffs are assigned weights between 10<sup>0</sup> and 10<sup>-4</sup> (default values), linearly, with all values below the lower cutoff getting assigned the minimum, and all values above the higher cutoff getting assigned the maximum weight.
  - Weights are expanded to the size of the input signal and median filtered (default filter length: 1999).
  - Signal is piecewise multiplied with weight vector.
  - Gaussian noise is added to the signal to mask low-RMS segments from voice detection (default sigma: 10<sup>-3</sup>)
- To change the filter just swap the ```rms_weighting_filter``` function with your own.
- The script expects speech audio files (wavs) following the naming convention of the noise reduction step (```pairPAIRNUMBER_LABNAME_SESSION_repaired_mono_noisered.wav``` ) in some folder. Sample call for `speech_rms_filtering.py` for pair 99, in lab Mordor:  
```python speech_rms_filtering 99 Mordor --audio_dir AUDIO_DIR```
- Standard output naming after RMS-based filtering follows this convention:  
```pairPAIRNUMBER_LABNAME_SESSION_repaired_mono_noisered_filtered.wav```  
(e.g. ```pair99_Mordor_freeConv_repaired_mono_noisered_filtered.wav```)
