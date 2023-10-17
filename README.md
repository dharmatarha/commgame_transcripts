# Transcription generation pipeline for CommGame experiment audio data (Hungarian)

Participants in the CommGame experiments have been recorded with HSE-150/SK stage microphones attached to IMG Stageline EMA-1 adapters, sampled by ESI Maya22 audio interfaces at 44100 Hz. Audio is stored in uncompressed 16-bit PCM format.  

For transcript generation, audio is first (1) preprocessed, then (2) filtered for background noise, optionally (3) further filtered in terms of overall power, then (4) segmented by a voice detector, and finally (5) transcribed by an ASR model.  
The pipeline relies on the following great projects:  

* [noisereduce package](https://github.com/timsainb/noisereduce)
* [Silero Voice Activity Detector](https://github.com/snakers4/silero-vad)
* [The BEA Speech Transcriber (BEAST2) model](https://phon.nytud.hu/bea/bea-base.html?lang=en)
  
The pipeline is prepared for Ubuntu 22.04 LTS as the target system. Matlab part has been tested with 2017a and above, python parts have been tested with 3.10.


### (1) Preprocessing raw audio  
**Matlab functions.**  
1. Audio is checked for missing segments (buffer underflow events during streaming) and deviations from nominal sampling rate (which is a strangely common problem), with resampling where necessary. This procedure reconstructs the true timeline of the recorded audio as best as we can do it offline. 
2. The above step is implemented in `audioRepair.m`, wrapped in `audioRepairWrapper.m` for batch processing. Sample call with our default parameters:
      ```audioRepair('/media/adamb/data_disk/CommGame/pair99', 99, 'freeConv', 0.020, 225, 0.5, 44100)```
4. The outputs are two mono wav files containing the preprocessed audio streams from the two labs. They are trimmed to start at the `sharedStartTime` timestamp and to end when the shorter of the two finishes.

### (2) Noise reduction  
**Python using [noisereduce](https://github.com/timsainb/noisereduce)**  

   Recordings were often sensitive enough to pick up not only speech from the participant wearing the microphone but also the other participant's speech streamed from the other lab and played from a speaker (crosstalk).  
   The noise reduction step aims to eliminate both crosstalk and the occasional line noise. Could be an optional step depending on the amount of crosstalk but has been used for all CommGame audio analysis so far.  
1. A csv file (`commgame_noiseclips.csv`) contains the start and end times of a noise (crosstalk) segment within each freeConv recording. First a noise clip (wav) is cut from each freeConv, that is, one for each participant using `noiseclip_generator.py`. Sample call for pair 99: ```python noiseclip_generator.py 99 --csv_path CSV_PATH```  
2. Might require parameter finetuning depending on the amount of cross-talk in the recording. Aggressive noise reduction degrades intelligibility of target speech stream. Generally, stationary reduction is more promising than non-stationary, as the latter tends to strongly degrade the target stream as well. Baseline params: prop_decr: 1.0, n_std_thresh: 1.0, chunk_size: 16384, n_fft: 1024.
