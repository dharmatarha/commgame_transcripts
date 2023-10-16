# Transcription generation pipeline for CommGame experiment audio data (Hungarian)

Participants in the CommGame experiments have been recorded with HSE-150/SK stage microphones attached to IMG Stageline EMA-1 adapters, sampled by ESI Maya22 audio interfaces at 44100 Hz. Audio is stored in uncompressed 16-bit PCM format.  

For transcript generation, audio is first (1) preprocessed, then (2) filtered for background noise, optionally (3) further filtered in terms of overall power, then (4) segmented by a voice detector, and finally (5) transcribed by an ASR model.  
The pipeline relies on the following projects:  
* [noisereduce package](https://github.com/timsainb/noisereduce)
* [Silero Voice Activity Detector](https://github.com/snakers4/silero-vad)
* [The BEA Speech Transcriber (BEAST2) model](https://phon.nytud.hu/bea/bea-base.html?lang=en)

### (1) Preprocessing raw audio
1. Audio is checked for missing segments (buffer underflow events during streaming) and deviations from nominal sampling rate (which is a strangely common problem), with resampling where necessary. This procedure reconstructs the true timeline of the recorded audio as best as we can do it offline. 
2. The above step is implemented in `audioRepair.m`, wrapped in `audioRepairWrapper.m` for batch processing. Sample call with our default parameters:
      ```audioRepair('/media/adamb/data_disk/CommGame/pair99', 99, 'freeConv', 0.020, 225, 0.5, 44100)```
4. The outputs are two mono wav files containing the preprocessed audio streams from the two labs. They are trimmed to start at the `sharedStartTime` timestamp and to end when the shorter of the two finishes.
