%{ 
Only core gmm_ubm trial, not loading any configs.

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


%% Step1: Training the UBM
%dataList = 'E:\temp\123\Smile\Lists\UBM.lst';

loadMem = true; %% load all UBM files into the memory

nmix = 64;
final_niter = 10;
ds_factor = 1;
if  ~loadMem || ~exist('dataUBM','var') 
    fid = fopen(dataList, 'rt');
    filenames = textscan(fid, '%q');
    fclose(fid);
    filenames = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       filenames, 'UniformOutput', false);

    %% сначала загружаем со всеми признаками
    featAll = 1:featMax;
	dataUBM = load_data(filenames{1});
    
end

%%removing not needed features
dataCut = dataUBM;
nfiles = size(dataCut, 1);
for ix = 1 : nfiles,
    dataCut{ix} = dataCut{ix}(featCol,:);
end

ubm = gmm_em(dataCut, nmix, final_niter, ds_factor, nworkers,ubmFile,featCol,vadCol,vadThr);


%% Step2: Adapting the speaker models from UBM
%fea_dir = 'E:\temp\123\Smile\';
%fea_ext = '.htk';
fid = fopen(trainList, 'rt');
C = textscan(fid, '%s %q');
fclose(fid);
model_ids = unique(C{1}, 'stable');
model_files = C{2};
nspks = length(model_ids);
map_tau = 10.0;
config = 'm';
gmm_models = cell(nspks, 1); 
for spk = 1 : nspks,
    ids = find(ismember(C{1}, model_ids{spk}));
    spk_files = model_files(ids);
    spk_files = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       spk_files, 'UniformOutput', false);
    gmm_models{spk} = mapAdapt(spk_files, ubm, map_tau, config,'',featCol,vadCol,vadThr);
end

%% Step3: Scoring the verification trials
%fea_dir = 'E:\temp\123\Smile\';
%fea_ext = '.htk';
%trial_list = 'E:\temp\123\Smile\Lists\Test.lst';
fid = fopen(testList, 'rt');
C = textscan(fid, '%s %q %s');
fclose(fid);
[model_ids, ~, Kmodel] = unique(C{1}, 'stable'); % check if the order is the same as above!
[test_files, ~, Ktest] = unique(C{2}, 'stable');
test_files = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       test_files, 'UniformOutput', false);
trials = [Kmodel, Ktest];
scores = score_gmm_trials(gmm_models, test_files, trials, ubm,featCol,vadCol,vadThr);

%% Step4: Computing the EER and plotting the DET curve
labels = C{3};
[eer, dcf1, dcf2] = compute_eer(scores, labels, true);
