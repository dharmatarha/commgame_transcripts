function audioRepairWrapper(pairList)
%% Wrapper for running audioRepair on a dataset
%
% Optional Input args for audioRepair are defined in the second code block 
% (hardcoded params). Input args "pairNo" and "inputDir" are derived from
% "pairList" and a hardcoded path. 
%
% For input arg "session", the function relies on vars "maxGame" from 
% "BGgames.mat" and "freeConvValid" from "freeConvValid.mat", which store 
% info about the available BG games and the validity of the freeConv 
% session for each pair.
%
% The function audioRepair is called in for loops for pairs and all
% possible sessions, including freeConv.
%
% Input:
% pairList  - Numeric vector, pair numbers.
%


%% Input check

if nargin ~= 1
    error('Input arg "pairList" is mandatory!');
end
if ~isvector(pairList) || ~isnumeric(pairList)
    error('Input arg "pairList" should be a numeric vector!');
end


%% Hardcoded params

% optional params for audioRepair.m
timeDiffThr = 0.02;
missingSampleThr = 225;
samplingTol = 0.5;
outputFs = 44100;

% path to data folders
basePath = '/media/adamb/data_disk/CommGame/';

% session types to iterate over
allSessions = {'freeConv', 'BG1', 'BG2', 'BG3', 'BG4', 'BG5', 'BG6', 'BG7', 'BG8'};


%% Start a diary file

c = clock;
logFile = ['audioRepairWrapperLog', num2str(c(2)), '_', num2str(c(3)), '_', num2str(c(4)), '_', num2str(c(5)), '.txt'];
diary(logFile);


%% Load BG games and freeConv info for pairs

tmp = load('/home/adamb/commgame_transcripts/preproc/BGgames.mat');
% check if we have BG info for the right pairs
bgInfoPairs = tmp.pairList;
if any(~ismember(pairList, bgInfoPairs))
    error('Cannot find BG game number info for at least one requested pair!');
end
% BG game info is stored in var "bgValid"
bgValid = tmp.bgValid;
% get the right bgValid rows for the elements of pairList
[~, i] = ismember(pairList, bgInfoPairs);
bgValid = bgValid(i, :);

% same for freeConv as for BG games
tmp = load('/home/adamb/commgame_transcripts/preproc/freeConvValid.mat');
freeConvInfoPairs = tmp.pairList;
if any(~ismember(pairList, freeConvInfoPairs))
    error('Cannot find freeConv validity info for at least one requested pair!');
end
freeConvValid = tmp.freeConvValid;
[~, i] = ismember(pairList, freeConvInfoPairs);
freeConvValid = freeConvValid(i);


%% Loops over pairs and sessions

pairIdx = 0;
for pairNo = pairList
    pairIdx = pairIdx + 1;
    
    % BG games validity for the pair
    bgs = bgValid(pairIdx, :);
    % freeConv validity for the pair
    fc = freeConvValid(pairIdx);
    
    % pair-specific arg inputDir for audioRepair based on basePath and
    % pairNo
    inputDir = [basePath, 'pair', num2str(pairNo), '/'];
    
    % sessions for given pair 
    counter = 1;
    if fc
        pairSessions = {'freeConv'};  % initiate sessions var with freeConv if there is valid freeConv for pair
        counter = counter + 1;
    else
        pairSessions = {};  % initiate it empty otherwise
    end
    for i = 1:size(bgs, 2)  % for each valid bg, add the right session name to pairSessions
        if bgs(i)
            pairSessions{counter} = ['BG', num2str(i)];
            counter = counter + 1;
        end
    end

    % log
    disp([char(10), char(10), char(10), 'Starting pair ', num2str(pairNo), '...']);
    disp(['BG games: ', num2str(bgs)]);
    disp(['freeConv validity: ', num2str(fc)]);
    disp(['Input dir for pair data: ', inputDir]);
    disp('Sessions list for pair: ');
    disp(pairSessions);
    disp([]);
    
    % loop over sessions
    for session = pairSessions
        currentSession = session{:};
        
        % the meat
        audioRepair(inputDir, pairNo, currentSession, timeDiffThr,... 
            missingSampleThr, samplingTol, outputFs);
        
    end  % for session
    
end  % for pairNo
        
        
diary off;






