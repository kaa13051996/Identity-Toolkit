%{ 

This is a demo on how to use the Identity Toolbox for i-vector based speaker
recognition. A relatively small scale task has been designed using speech
material from the TIMIT corpus. There are a total of 630 (192 female and
438 male) speakers in TIMIT, from which we have selected 530 speakers for
background modeBl training and the remaining 100 (30 female and 70 male)
speakers are used for tests. There are 10 short sentences per speaker in
TIMIT. For background model training we use all sentences from all 530
speakers (i.e., 5300 speech recordings in total). For speaker specific
model training we use 9 out of 10 sentences per speaker and keep the
remaining 1 sentence for tests. Verification trials consist of all possible
model-test combinations, making a total of 10,000 trials (100 target vs
9900 impostor trials).

Assuming that audio recordings are already converted 
into cepstral features, there are 5 steps involved:
 
 1. training a UBM from background data
 2. learning a total variability subspace from background statistics
 3. training a Gaussian PLDA model with development i-vectors
 4. scoring verification trials with model and test i-vectors
 5. computing the performance measures (e.g., EER)

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
nworkers = 12;
nworkers = min(nworkers, feature('NumCores'));
p = gcp('nocreate'); 
if isempty(p), parpool(nworkers); end 

%% Loading config file
fea_dir =  'E:\temp\123\Smile\'; % Feature files list
configDir = 'E:\temp\123\data\ListsMF\';
ini = IniConfig();
ini.ReadFile(strcat(configDir,'cMFC.lst'));
sections = ini.GetSections();
indGMM = find(ismember(sections,'[UBM GMM]'));
indDS = find(ismember(sections,'[Data selection]'));
indIV = find(ismember(sections,'[I-vector]'));
keys=ini.GetKeys(sections{indGMM});
ubmMap = containers.Map(keys,ini.GetValues(sections{indGMM}, keys)); % reading UBM GMM section
dataList = strcat(configDir,ubmMap('ubmList')); %UBM training list
trainList = strcat(configDir,ubmMap('trainList')); % Speaker modelling list
testList = strcat(configDir,ubmMap('testList')); % Trials list
ubmFile=strcat(configDir,ubmMap('ubmFile')); %ubmFile=ubmFile{1};

keys=ini.GetKeys(sections{indDS}); % reading Data Selection section
dataMap = containers.Map(keys,ini.GetValues(sections{indDS}, keys));
featCol=str2num(dataMap('columns'));
vadCol=dataMap('vadColumn');
vadThr=dataMap('vadThreshold');

keys=ini.GetKeys(sections{indIV});
ivMap = containers.Map(keys,ini.GetValues(sections{indIV}, keys)); % reading I-vector section
ubmIdList = strcat(configDir,ivMap('ubmIdList')); %UBM id training list
bwFile=strcat(configDir,ivMap('bwFile')); 
tFile=strcat(configDir,ivMap('tFile')); 
pldaFile=strcat(configDir,ivMap('pldaFile'));


%% Step1: Training the UBM
%dataList = 'E:\temp\123\Smile\Lists\UBM.lst';
%Check if UBM is already trained
if exist(ubmFile,'file')
      
    load(ubmFile);
    ubm = gmm;
    clear('gmm');
else    
nmix        = 256;
final_niter = 10;
ds_factor   = 1;
fid = fopen(dataList, 'rt');
filenames = textscan(fid, '%q');
fclose(fid);
filenames = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       filenames, 'UniformOutput', false);
ubm = gmm_em(filenames{1}, nmix, final_niter, ds_factor, nworkers,ubmFile,featCol,vadCol,vadThr);
end

%% Step2: Learning the total variability subspace from background data
tv_dim = 400; 
niter  = 5;
%dataList = 'E:\temp\123\Smile\Lists\UBM.lst';
fid = fopen(dataList, 'rt');
C = textscan(fid, '%q');
fclose(fid);
C = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       C, 'UniformOutput', false);
feaFiles = C{1};

if exist(bwFile,'file')
      
    load(bwFile);

else    
stats = cell(length(feaFiles), 1);

parfor file = 1 : length(feaFiles),
    [N, F] = compute_bw_stats(feaFiles{file}, ubm, featCol,vadCol,vadThr);
    stats{file} = [N; F];
end
save(bwFile,'stats');

end
if exist(tFile,'file')
      
    load(tFile);
    %ubm = gmm;
    %clear('gmm');
else    
T = train_tv_space(stats, ubm, tv_dim, niter, nworkers);
save(tFile,'T');
end

%% Step3: Training the Gaussian PLDA model with development i-vectors

if exist(pldaFile,'file')
      
    load(pldaFile);

else
lda_dim = 200;

nphi    = 200;
niter   = 10;
%dataList = 'E:\temp\123\Smile\Lists\UBM_id.lst';
fid = fopen(ubmIdList, 'rt');
C = textscan(fid, '%q %s');
fclose(fid);
feaFiles = C{1};
dev_ivs = zeros(tv_dim, length(feaFiles));
parfor file = 1 : length(feaFiles),
    dev_ivs(:, file) = extract_ivector(stats{file}, ubm, T);
end
% reduce the dimensionality with LDA
spk_labs = C{2};
nSpeakers = unique(C{2}, 'stable');
nSpeakers = size(nSpeakers,1);

lda_dim = min(lda_dim, nSpeakers-1);
V = lda(dev_ivs, spk_labs);
dev_ivs = V(:, 1 : lda_dim)' * dev_ivs;
%------------------------------------
plda = gplda_em(dev_ivs, spk_labs, nphi, niter);

save(pldaFile,'plda','V','lda_dim');
end

%% Step4: Scoring the verification trials
%fea_dir = 'E:\temp\123\Smile\';
%fea_ext = '.htk';
fid = fopen(trainList, 'rt');
C = textscan(fid, '%s %q');
fclose(fid);
model_ids = unique(C{1}, 'stable');
model_files = C{2};
nspks = length(model_ids);
model_ivs1 = zeros(tv_dim, nspks);
model_ivs2 = model_ivs1;
parfor spk = 1 : nspks,
    ids = find(ismember(C{1}, model_ids{spk}));
    spk_files = model_files(ids);
    spk_files = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       spk_files, 'UniformOutput', false);
    N = 0; F = 0; 
    for ix = 1 : length(spk_files),
        [n, f] = compute_bw_stats(spk_files{ix}, ubm, featCol, vadCol, vadThr);
        N = N + n; F = f + F; 
        model_ivs1(:, spk) = model_ivs1(:, spk) + extract_ivector([n; f], ubm, T);
    end
    model_ivs2(:, spk) = extract_ivector([N; F]/length(spk_files), ubm, T); % stats averaging!
    model_ivs1(:, spk) = model_ivs1(:, spk)/length(spk_files); % i-vector averaging!
end
%trial_list = 'E:\temp\123\Smile\Lists\Test.lst';
fid = fopen(testList, 'rt');
C = textscan(fid, '%s %q %s');
fclose(fid);
[model_ids, ~, Kmodel] = unique(C{1}, 'stable'); % check if the order is the same as above!
[test_files, ~, Ktest] = unique(C{2}, 'stable');
test_files = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       test_files, 'UniformOutput', false);
test_ivs = zeros(tv_dim, length(test_files));
parfor tst = 1 : length(test_files),
    [N, F] = compute_bw_stats(test_files{tst}, ubm, featCol, vadCol, vadThr);
    test_ivs(:, tst) = extract_ivector([N; F], ubm, T);
end
% reduce the dimensionality with LDA
model_ivs1 = V(:, 1 : lda_dim)' * model_ivs1;
model_ivs2 = V(:, 1 : lda_dim)' * model_ivs2;
test_ivs = V(:, 1 : lda_dim)' * test_ivs;
%------------------------------------
scores1 = score_gplda_trials(plda, model_ivs1, test_ivs);
linearInd =sub2ind([nspks, length(test_files)], Kmodel, Ktest);
scores1 = scores1(linearInd); % select the valid trials

scores2 = score_gplda_trials(plda, model_ivs2, test_ivs);
scores2 = scores2(linearInd); % select the valid trials

%% Step5: Computing the EER and plotting the DET curve
labels = C{3};
eer1 = compute_eer(scores1(linearInd), labels, true); % IV averaging
hold on
eer2 = compute_eer(scores2(linearInd), labels, true); % stats averaging
