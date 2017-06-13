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

loadMem = true; %% load all files into the memory
loadUBM = false; %% load UBM from disk

if loadUBM&&exist(ubmFile,'file')
  
load(ubmFile);
ubm = gmm;
clear('gmm');

else    
nmix = 256;
final_niter = 10;
ds_factor = 1;
if  ~loadMem || ~exist('dataUBM','var') 
    
    %dataUBM = load_data(filenames{1});
    %load('E:\temp\123\data\DBN\ubm3.mat');
    load('E:\temp\123\data\DBN\ubm1.mat');
    dataCut=out1';
    
end

%%removing not needed features
%dataCut = dataUBM;
%nfiles = size(dataCut, 1);
%for ix = 1 : nfiles,
%    dataCut{ix} = dataCut{ix}(featCol,:);
%end

% тут ставим вычисления на GPU
%ubm = gmm_em_operations(dataCut, nmix, final_niter, ds_factor, nworkers,'',featCol,vadCol,vadThr);
%ubm = gmm_em_gpu_parf(dataCut, nmix, final_niter, ds_factor, nworkers,'',featCol,vadCol,vadThr);
%ubm = gmm_em_gpu(dataCut, nmix, final_niter, ds_factor, nworkers,'',featCol,vadCol,vadThr);
ubm = gmm_em(dataCut, nmix, final_niter, ds_factor, nworkers,'',featCol,vadCol,vadThr);

%ubm = gmm_em_gpu_trans(dataCut, nmix, final_niter, ds_factor, nworkers,'',featCol,vadCol,vadThr);
%ubm = gmm_em_gpu_mem_test(dataCut, nmix, final_niter, ds_factor, nworkers,'',featCol,vadCol,vadThr);

%ubm = gmm_em(dataCut, nmix, final_niter, ds_factor, nworkers,'',featCol,vadCol,vadThr);

end

%% Step2: Adapting the speaker models from UBM
%fea_dir = 'E:\temp\123\Smile\';
%fea_ext = '.htk';
map_tau = 10.0;
config = 'm';

if  ~loadMem || ~exist('dataTrain','var') 
    
    %load('E:\temp\123\data\DBN\sp3.mat');
    load('E:\temp\123\data\DBN\sp1.mat');
    dataCut=out1;
    nspks = length(dataCut);
    gmm_models = cell(nspks, 1); 
end

%%вырезаем ненужные фичи
%dataCut = dataTrain;
%for spk = 1 : nspks,
%    for spk2 = 1: length(dataTrain{spk})
%        dataCut{spk}{spk2} = dataCut{spk}{spk2}(featCol,:);
%    end
%end
%%тут уже с загруженными данными проводим адаптацию
for spk = 1 : nspks,
    gmm_models{spk} = mapAdapt(dataCut{spk}, ubm, map_tau, config,'',featCol,vadCol,vadThr);
end


%% Step3: Scoring the verification trials
%fea_dir = 'E:\temp\123\Smile\';
%fea_ext = '.htk';
%trial_list = 'E:\temp\123\Smile\Lists\Test.lst';

if  ~loadMem || ~exist('dataTest','var') 
   
    %load('E:\temp\123\data\DBN\tst3.mat');
    load('E:\temp\123\data\DBN\tst1.mat');
    load('E:\temp\123\data\DBN\labels.mat');
    dataCut=out1';

    
end

%%removing not needed features
%dataCut = dataTest;
%nfiles = length(dataCut);
%for ix = 1 : nfiles,
%    dataCut{ix} = dataCut{ix}(featCol,:);
%end


%%итого подгрузили в память все тесты, чтобы каждый раз не читать с диска

scores = score_gmm_trials(gmm_models, dataCut, trials, ubm,featCol,vadCol,vadThr);

%% Step4: Computing the EER and plotting the DET curve

[eer, dcf1, dcf2] = compute_eer(scores, labels, true);
