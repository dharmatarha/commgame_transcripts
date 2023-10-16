# commgame_transcripts
Transcription generation pipeline for CommGame experiment audio data (Hungarian)

###Preprocessing raw audio
1. Audio is checked for missing segments (buffer underflow events during streaming) and deviations from nominal sampling rate (a common problem…), with resampling where necessary. This procedure reconstructs the true timeline of the recorded audio as best as we can do it offline. 
2. The above step is implemented in “audioRepair.m” in the “rater_task” repo, wrapped in “audioRepairWrapper.m” for batch processing. Sample call with employed parameters: audioRepair(‘/media/adamb/data_disk/CommGame/pair99’, 99, ‘freeConv’, 0.020, 225, 0.5, 44100)
3. The outputs are two mono wav files containing the preprocessed audios from the two labs. They are trimmed to start at “sharedStartTime” and to end when the shorter of the two ends.
