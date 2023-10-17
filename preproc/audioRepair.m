function audioRepair(inputDir, pairNo, session, timeDiffThr, missingSampleThr, samplingTol, outputFs)
%% Function to repair buffer underflow errors and bad sampling rates
%
% USAGE: audioRepair(inputDir,
%                    pairNo,
%                    session,
%                    timeDiffThr=0.020,
%                    missingSampleThr=225,
%                    samplingTol=0.5,
%                    outputFs=44100)
%
% Two types of audio recording problems are mitigated by the function:
%
% (1) When a buffer underflow occured, we see the details of the missing
% portion from the audio status parameters saved out during the task. Such
% missing segments are recovered (injected) as segments filled with silence.
% This behavior is controlled by input args "timeDiffThr" and 
% "missingSampleThr".
%
% (2) Sampling rates are not fully consistent across different sound cards
% used and might show deviations from nominal sampling rate. Such problems
% are detected and the recorded audio resampled if necessary. The maximum
% tolerated deviation from nominal sampling rate is controlled by
% input arg "samplingTol". 
%
% Audio recordings are also aligned to a common start time
% "sharedStartTime" loaded from recording-specific .mat files.
%
% Mandatory inputs:
% inputDir   - Char array, path to folder holding pair-level data. The
%              folder is searched recursively for the right files.
% pairNo     - Numeric value, pair number, one of 1:99.
% session    - Char array, session type, one of 
%              {'freeConv', 'BG1', 'BG2', ..., 'BG8'}.
% 
% Optional inputs:
% timeDiffThr      - Time difference threshold between subsequent audio 
%                    packets in seconds. If the recording times of
%                    subsequent packets  differ more than this threshold,
%                    there could have been a buffer underflow event, and 
%                    the packets are flagged for a further check based on 
%                    "missingSampleThr". Defaults to 0.02 (20 msec),
%                    roughly double the "normal" audio packet size.
% missingSampleThr - Threshold for the number of "missing" audio frames
%                    after a temporal deviation (time difference, see 
%                    "timeDiffThr") is detected. If the threshold is reached,
%                    a silent (zero-filled) segment is inserted for the 
%                    missing segment. Defaults to 225, corrresponding to
%                    missing data of 5 msec at 44.1 kHz.
% samplingTol      - Tolerance for deviation from nominal sampling rate
%                    (see "fs") in Hz, defaults to 0.5.
% outputFs         - Sampling rate in Hz for output files (wavs). Defaults to 44100.
%
% The outputs are the edited, synched audio files at:
% inputDir/pair[pairNo]_Mordor_[session]_audio_repaired.wav
% inputDir/pair[pairNo]_Gondor_[session]_audio_repaired.wav
%
%
% 2023.05.
%
% Notes:
% - Current implementation depends on rigid file structure for audio data.
% - Input files are expected to have a "nominal" 44 kHz or 44.1 kHz sampling
% freq.
% - Original version only worked on 'freeConv' data, full functionality is
% added in 2023.06.
% - Final bug fixes in 2023.08.
%


%% Input checks

if ~ismember(nargin, 3:7)
    error('Input args inputDir, pairNo and sesison are required while timeDiffThr, missingSampleThr, samplingTol and outputFs are optional!');
end
if nargin < 7 || isempty(outputFs)
    outputFs = 44100;
end
if nargin < 6 || isempty(samplingTol)
    samplingTol = 0.5;
end
if nargin < 5 || isempty(missingSampleThr)
    missingSampleThr = 225;
end
if nargin < 4 || isempty(timeDiffThr)
    timeDiffThr = 0.02;
end

% check mandatory inputs
if ~exist(inputDir, 'dir')
    error('Input arg "inputDir" is not a folder!');
end
if ~ismember(pairNo, 1:999)
    error('Input arg "pairNo" should be one of 1:999!');
end
if ~ismember(session, {'freeConv', 'BG1', 'BG2', 'BG3', 'BG4', 'BG5', 'BG6', 'BG7', 'BG8'})
    error('Input arg "session" should be one of {freeConv, BG1, BG2, ..., BG8}!');
end

disp([char(10), 'Called audioRepair with input args:',...
    char(10), 'Input dir: ', inputDir, ...
    char(10), 'Pair number: ', num2str(pairNo), ...
    char(10), 'Session type: ', session, ...
    char(10), 'Time difference threshold: ', num2str(timeDiffThr*1000), ' ms', ...
    char(10), 'Missing sample threshold: ', num2str(missingSampleThr), ' frames', ...
    char(10), 'Sampling rate deviation tolerance: ', num2str(samplingTol), ' Hz', ...
    char(10), 'Nominal sampling rate: ', num2str(outputFs), ' Hz']);


%% Hardcoded params

% input wavs are expected to have one of these sampling frequencies  
inputFs = [44000, 44100]; 
startTol = 0.0001;  % tolerance for shared start time difference across labs


%% Find pair-specific .mat and .wav files

% audio files
audioFiles = struct;
for labIdx = {'mordor', 'gondor'}
    lab = labIdx{:};
    % for file path definitions, first char is uppercase
    labName = lab; labName(1) = upper(labName(1));
    % find files with dir
    tmpwav = dir([inputDir, 'pair', num2str(pairNo), '_', labName,... 
        '/pair', num2str(pairNo), '_', labName, '_behav/pair',... 
        num2str(pairNo), '_', labName, '_', session, '_audio.wav']);
    tmpmat = dir([inputDir, 'pair', num2str(pairNo), '_', labName,... 
        '/pair', num2str(pairNo), '_', labName, '_behav/pair',... 
        num2str(pairNo), '_', labName, '_', session, '_audio.mat']);
    % sanity check - is there only one?
    if length(tmpwav) ~= 1 || length(tmpmat) ~= 1 
        error(['Found none or more than one audio wav or mat file for ', labName, ', for ', session, '!']);
    end
    % store full path    
    audioFiles.(lab).wav = fullfile(tmpwav(1).folder, tmpwav(1).name);
    audioFiles.(lab).mat = fullfile(tmpmat(1).folder, tmpmat(1).name);
end
    
% video files (for shared start time)
videoFiles = struct;
for labIdx = {'mordor', 'gondor'}
    lab = labIdx{:};
    % for file path definitions, first char is uppercase
    labName = lab; labName(1) = upper(labName(1));
    % find file with dir, name depends on session type
    if strcmp(session, 'freeConv')
        tmpmat = dir([inputDir, 'pair', num2str(pairNo), '_', labName,... 
            '/pair', num2str(pairNo), '_', labName, '_behav/pair',... 
            num2str(pairNo), '_', labName, '_', session, '_videoTimes.mat']);
    else
        tmpmat = dir([inputDir, 'pair', num2str(pairNo), '_', labName,... 
            '/pair', num2str(pairNo), '_', labName, '_behav/pair',... 
            num2str(pairNo), '_', labName, '_', session, '_times.mat']);
    end
    % sanity check - is there only one?
    if length(tmpmat) ~= 1 
        error(['Found none / more than one video mat file for ', labName, ', for ', session, '!']);
    end
    % store full path    
    videoFiles.(lab).mat = fullfile(tmpmat(1).folder, tmpmat(1).name);
end

disp('Found relevant files:');
disp(audioFiles.mordor); disp(audioFiles.gondor);
disp(videoFiles.mordor); disp(videoFiles.gondor);


%% Extract all relevant timestamps

% VIDEO
% Get shared start time
for labIdx = {'mordor', 'gondor'}
    lab = labIdx{:};
    tmp = load(videoFiles.(lab).mat);
    sharedStartTime.(lab) = tmp.sharedStartTime;
end
if sharedStartTime.mordor ~= sharedStartTime.gondor && abs(sharedStartTime.mordor - sharedStartTime.gondor > startTol)
    error('Shared start times do not match across labs! Check which files are used? Synch problem?');
else
    sharedStartTime = sharedStartTime.mordor;
end

% AUDIO
% timestamps of first recorded audio frames + frame data
for labIdx = {'mordor', 'gondor'}
    lab = labIdx{:};
    tmp = load(audioFiles.(lab).mat);
    audioStart.(lab) = tmp.perf.firstFrameTiming;
    tstats.(lab) = tmp.perf.tstats;
end

% sanity check - audio recordings must have started before video stream
if audioStart.mordor >= sharedStartTime || audioStart.gondor >= sharedStartTime
    error('Insane audio versus task and video start times!');
end

disp('Extracted relevant timestamps and audio recording metadata');


%% Interim estimation of sampling frequency
% The aim of this segment is to estimate the "empirical" or "real" sampling
% rate of the audio recording, as that is usually a bit different than what
% was set (due to audio card problems...).
% To this end, we first find a long segment of frame data where there was
% no underflow event (see above), then simply match the elapsed time with
% the number of samples.

fsEmp = struct;
suspectFrames = struct; suspectFrames.mordor = []; suspectFrames.gondor = [];
audioTimes = struct; audioTimes.mordor = []; audioTimes.gondor = [];
elapsedSamples = struct; elapsedSamples.mordor = []; elapsedSamples.gondor = [];

for labIdx = {'mordor', 'gondor'}
    lab = labIdx{:};
    
    % extract timing and samples data for each frame from "tstats", 
    % find frames with suspiciously large temporal gap
    audioTimes.(lab) = tstats.(lab)(2, :)';
    elapsedSamples.(lab) = tstats.(lab)(1, :)';
    suspectFrames.(lab) = find(diff(audioTimes.(lab)) > timeDiffThr);  
    
    % if there are suspicious frames, we have to find a relatively long 
    % segment between them to estimate sampling freq 
    if ~isempty(suspectFrames.(lab))
        audioEvents = sort([1, size(tstats.(lab), 2), suspectFrames.(lab)(:, 1)'+1]);
        longSegmentStart = find(diff(audioEvents) == max(diff(audioEvents)));
        segmentBounds = [audioEvents(longSegmentStart), audioEvents(longSegmentStart + 1)];
        tmpTimes = [tstats.(lab)(2, segmentBounds(1)), tstats.(lab)(2, segmentBounds(2))];
        tmpSamples = [tstats.(lab)(1, segmentBounds(1)), tstats.(lab)(1, segmentBounds(2))];
        fsEmp.(lab) = diff(tmpSamples)/diff(tmpTimes);
        
    % if there are no suspicious frames, estimation is a piece of cake    
    else
        tmpTimes = [tstats.(lab)(2, 1), tstats.(lab)(2, end)];
        tmpSamples = [tstats.(lab)(1, 1), tstats.(lab)(1, end)];
        fsEmp.(lab) = diff(tmpSamples)/diff(tmpTimes);
        
    end  % if ~isempty
    
end  % for labIdx

disp('Used audio frame data to estimate real sampling rates:');
disp(['For Mordor, sampling rate was ', num2str(fsEmp.mordor), ' Hz']);
disp(['For Gondor, sampling rate was ', num2str(fsEmp.gondor), ' Hz']);


%% Find underflows in audio channels, based on audio frame timing

% Correct for missing audio packets (occasional underflows) that 
% correspond to jumps in stream timings without audio data.
% First, detect "jumps", that is, audio frames where there is a 
% "large" change in streaming time from frame to frame, while the number of 
% elapsed samples does not match it.
% Then later we will inject sufficiently long "silence" frames/segments to 
% account for the missing data - see below, after loading the audio.

frames2Repair = struct;
frames2Repair.mordor = [];
frames2Repair.gondor = [];
for labIdx = {'mordor', 'gondor'}
    lab = labIdx{:};
    
    % counter for frames requiring "repair"
    counter = 1;
    
    % check each suspect audioframe for skipped material
    if ~isempty(suspectFrames.(lab))
        
        for i = 1:length(suspectFrames.(lab))
            timingDiff = audioTimes.(lab)(suspectFrames.(lab)(i)+1) - audioTimes.(lab)(suspectFrames.(lab)(i));
            sampleDiff = elapsedSamples.(lab)(suspectFrames.(lab)(i)+1) - elapsedSamples.(lab)(suspectFrames.(lab)(i));
            expectedSamples = timingDiff*fsEmp.(lab);
            if expectedSamples - sampleDiff > missingSampleThr
               frames2Repair.(lab)(counter, 1:2) = [suspectFrames.(lab)(i), expectedSamples-sampleDiff];
               counter = counter + 1;
            end
        end  % for i
        
    end  % if ~isempty 
    
end  % for lab

disp('Checked for missing samples (underflows)');
disp(['For Mordor, there were ', num2str(size(frames2Repair.mordor, 1)), ' suspected events']);
disp(['For Gondor, there were ', num2str(size(frames2Repair.gondor, 1)), ' suspected events']);


%% Load audio

audioData = struct;
for labIdx = {'mordor', 'gondor'}
    lab = labIdx{:};
    [audioData.(lab), tmpFs] = audioread(audioFiles.(lab).wav); 
    if ~ismember(tmpFs, inputFs)
        error(['Unexpected sampling freq (', num2str(tmpFs), ') in audio file at ', audioFiles.(lab).wav ]);
    end
end

disp('Loaded audio files');


%% Repair loaded audio for missing frames (underflows)

for labIdx = {'mordor', 'gondor'}
    lab = labIdx{:};
    
    if ~isempty(frames2Repair.(lab))
        elapsedSamples = tstats.(lab)(1, :)';
        
        % for inserting audio samples, do it in reverse order, otherwise 
        % the indices get screwed
        for i = size(frames2Repair.(lab), 1):-1:1
            % sample to insert silence at
            startSample = elapsedSamples(frames2Repair.(lab)(i, 1) + 1);
            % define silence (zeros)
            silentFrame = zeros(round(frames2Repair.(lab)(i, 2)), 2);
            % special rule for inserting silent frames when those would be at the very end, 
            % potentially out of bounds of recorded audio
            if startSample > size(audioData.(lab), 1) + 1
                audioData.(lab) = [audioData.(lab); silentFrame];
            % otherwise we insert silent frames to their expected location
            else
                audioData.(lab) = [audioData.(lab)(1:startSample, 1:2); silentFrame; audioData.(lab)(startSample+1:end, 1:2)];
            end
        end  % for i
        
    end  % if ~isempty
    
end  % for lab

disp('Inserted silent frames for detected underflow events');


%% Estimate real (empirical) sampling frequency from the full length, repaired audio
% estimate sampling frequency based on the size of the (repaired) audio
% data and the total time elapsed while recording

fsEmpFinal = struct;
totalTime = struct;
for labIdx = {'mordor', 'gondor'}
    lab = labIdx{:};
    
    streamTimes = tstats.(lab)(2, :)';
    totalSamples =size(audioData.(lab), 1);
    totalTime.(lab) = streamTimes(end)-streamTimes(1);
    fsEmpFinal.(lab) = totalSamples/totalTime.(lab);
    
    disp(['Estimated real sampling frequency for ', lab, ' audio, based on all audio data: ',... 
          num2str(fsEmpFinal.(lab)), ' Hz']);
      
end


%% Resample audio channels, if needed

for labIdx = {'mordor', 'gondor'}
    lab = labIdx{:};
    
    if abs(fsEmpFinal.(lab) - outputFs) > samplingTol
        tx = 0 : 1/fsEmpFinal.(lab) : totalTime.(lab);
        data = audioData.(lab);
        
        % due to numeric errors there could be a slight mismatch between audio
        % frames and corresponding timestamps - check for discrepancy
        if numel(tx) ~= size(data, 1)
            % report if the difference is too large
            if abs(numel(tx) ~= size(data, 1)) > 2
                disp(['WARNING! At the resampling step for ', lab, ', audio data size is ',... 
                    num2str(size(data, 1)), ' while estimated time points is a vector of length ',... 
                    num2str(numel(tx)), '!']);
            end
            tx = tx(1:size(data, 1));
        end
        audioData.(lab) = resample(data, tx, outputFs);
        disp(['Resampled ', lab, ' audio to nominal (', num2str(outputFs),... 
            ' Hz) sampling frequency']);
        
    end  % if

end  % for labIdx


%% Edit audio to common start:
% Both channels are trimmed so that they start from sharedStartTime and end
% when the shorter of the two audio recordings ended.
% Since sampling frequency issues are already fixed at this point, we
% assume that sampling frequency = outputFs, and use that for trimming

% trim from start and end
for labIdx = {'mordor', 'gondor'}
    lab = labIdx{:};
    startDiff = sharedStartTime - audioStart.(lab);
    audioData.(lab) = audioData.(lab)(round(startDiff*outputFs)+1 : end, :);
end
disp('Trimmed both audio channels to video start');

% turn to mono and normalize intensity
for labIdx = {'mordor', 'gondor'}
    lab = labIdx{:};
    audioData.(lab) = mean(audioData.(lab), 2);
    audioData.(lab) = (audioData.(lab) / max(audioData.(lab))) * 0.99;
end
disp('Audio channels are set to mono and normalized');

% check length, there might be a difference still
if length(audioData.mordor) ~= length(audioData.gondor)
    lm = length(audioData.mordor);
    lg = length(audioData.gondor);
    if lm < lg
        audioData.gondor = audioData.gondor(1:lm);
    elseif lm > lg
        audioData.mordor = audioData.mordor(1:lg);
    end
    disp('Audio channel length values adjusted (trimmed to the shorter)');
    disp('Original length for Mordor and Gondor were, respectively: ');
    disp([lm, lg]./outputFs);
end


%% save audio files

% output paths
outputAudioMordor = fullfile(inputDir, ['pair', num2str(pairNo), '_Mordor_', session, '_repaired_mono.wav']);
outputAudioGondor = fullfile(inputDir, ['pair', num2str(pairNo), '_Gondor_', session, '_repaired_mono.wav']);

audiowrite(outputAudioMordor, audioData.mordor, outputFs);
disp('Mordor audio saved out to:');
disp(outputAudioMordor);
audiowrite(outputAudioGondor, audioData.gondor, outputFs);
disp('Gondor audio saved out to:');
disp(outputAudioGondor);

return

