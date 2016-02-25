%{ 

This is a demo on how to use the Identity Toolbox for GMM-UBM based speaker
recognition. A relatively small scale task has been designed using speech
material from the TIMIT corpus. There are a total of 630 (192 female and
438 male) speakers in TIMIT, from which we have selected 530 speakers for
background model training and the remaining 100 (30 female and 70 male)
speakers are used for tests. There are 10 short sentences per speaker in
TIMIT. For background model training we use all sentences from all 530
speakers (i.e., 5300 speech recordings in total). For speaker specific
model training we use 9 out of 10 sentences per speaker and keep the
remaining 1 sentence for tests. Verification trials consist of all possible
model-test combinations, making a total of 10,000 trials (100 target vs
9900 impostor trials).

Assuming that audio recordings are already converted 
into cepstral features, there are 4 steps involved:
 
 1. training a UBM from background data
 2. MAP adapting speaker models from the UBM using enrollment data
 3. scoring verification trials
 4. computing the performance measures (e.g., EER)

Note: given the relatively small size of the task, we can load all the data 
and models into memory. This, however, may not be practical for large scale 
tasks (or on machines with a limited memory). In such cases, the parameters 
should be saved to the disk.

Omid Sadjadi <s.omid.sadjadi@gmail.com>
Microsoft Research, Conversational Systems Research Center

%}

clc
clear

%% Step0: Opening MATLAB pool
nworkers = 8;
nworkers = min(nworkers, feature('NumCores'));

p = gcp('nocreate'); 
if isempty(p), parpool(nworkers); end 

%% Loading config file
fea_dir =  'E:\temp\123\Smile\'; % Feature files list
%fea_dir =  'E:\temp\123\mftNK\'; % Feature files list
%fea_dir =  'E:\temp\123\data\12MFC+LPC+E+MFT3\'; % Feature files list
%fea_dir =  'E:\temp\123\data\12MFC+MFT3\'; % Feature files list
%configDir = strcat(fea_dir,'Lists\');
configDir = 'E:\temp\123\data\ListsF\';
fid = fopen(strcat(configDir,'cMFC_F0.lst'), 'rt');
C = textscan(fid, '%q');
fclose(fid);
dataList = strcat(configDir,C{1}(1)); %UBM training list
trainList = strcat(configDir,C{1}(3)); % Speaker modelling list
testList = strcat(configDir,C{1}(4)); % Trials list
ubmFile=strcat(configDir,C{1}(5)); ubmFile=ubmFile{1};
featCol=str2num(C{1}{6});


%% Step1: Training the UBM
%dataList = 'E:\temp\123\Smile\Lists\UBM.lst';
%Check if UBM is already trained
if exist(ubmFile,'file')
  
load(ubmFile);
ubm = gmm;
clear('gmm');

else    
nmix = 256;
final_niter = 10;
ds_factor = 1;
fid = fopen(dataList{1}, 'rt');
filenames = textscan(fid, '%q');
fclose(fid);
filenames = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       filenames, 'UniformOutput', false);
ubm = gmm_em(filenames{1}, nmix, final_niter, ds_factor, nworkers,ubmFile,featCol);
end

%% Step2: Adapting the speaker models from UBM
%fea_dir = 'E:\temp\123\Smile\';
%fea_ext = '.htk';
fid = fopen(trainList{1}, 'rt');
C = textscan(fid, '%s %q');
fclose(fid);
model_ids = unique(C{1}, 'stable');
model_files = C{2};
nspks = length(model_ids);
map_tau = 10.0;
config = 'mwv';
gmm_models = cell(nspks, 1); 
for spk = 1 : nspks,
    ids = find(ismember(C{1}, model_ids{spk}));
    spk_files = model_files(ids);
    spk_files = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       spk_files, 'UniformOutput', false);
    gmm_models{spk} = mapAdapt(spk_files, ubm, map_tau, config,'',featCol);
end

%% Step3: Scoring the verification trials
%fea_dir = 'E:\temp\123\Smile\';
%fea_ext = '.htk';
%trial_list = 'E:\temp\123\Smile\Lists\Test.lst';
fid = fopen(testList{1}, 'rt');
C = textscan(fid, '%s %q %s');
fclose(fid);
[model_ids, ~, Kmodel] = unique(C{1}, 'stable'); % check if the order is the same as above!
[test_files, ~, Ktest] = unique(C{2}, 'stable');
test_files = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       test_files, 'UniformOutput', false);
trials = [Kmodel, Ktest];
scores = score_gmm_trials(gmm_models, test_files, trials, ubm,featCol);

%% Step4: Computing the EER and plotting the DET curve
labels = C{3};
[eer, dcf1, dcf2] = compute_eer(scores, labels, true);
