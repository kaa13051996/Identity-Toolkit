
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
    fid = fopen(dataList, 'rt');
    filenames = textscan(fid, '%q');
    fclose(fid);
    filenames = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       filenames, 'UniformOutput', false);

    %% сначала загружаем со всеми признаками
    %%featAll = 1:featMax;
    dataUBM = load_data_gmm(filenames{1});

end

%%removing not needed features
dataCut = dataUBM;
nfiles = size(dataCut, 1);
for ix = 1 : nfiles,
    dataCut{ix} = dataCut{ix}(featCol,:);
end

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
fid = fopen(trainList, 'rt');
C = textscan(fid, '%s %q');
fclose(fid);
model_ids = unique(C{1}, 'stable');
model_files = C{2};
nspks = length(model_ids);
gmm_models = cell(nspks, 1); 
dataTrain = cell(nspks,1);

for spk = 1 : nspks,
    ids = find(ismember(C{1}, model_ids{spk}));
    spk_files = model_files(ids);
    spk_files = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       spk_files, 'UniformOutput', false);
    %%загружаем обучающие данные
    dataTrain{spk} = load_data_gmm(spk_files);
end
end

%%вырезаем ненужные фичи
dataCut = dataTrain;
for spk = 1 : nspks,
    for spk2 = 1: length(dataTrain{spk})
        dataCut{spk}{spk2} = dataCut{spk}{spk2}(featCol,:);
    end
end
%%тут уже с загруженными данными проводим адаптацию
for spk = 1 : nspks,
    gmm_models{spk} = mapAdapt(dataCut{spk}, ubm, map_tau, config,'',featCol,vadCol,vadThr);
end
% 
% scores_learning = zeros(size(gmm_models,1),1);        
% for i=1:size(gmm_models,1)      
%     [~,score] = predict(gmm_models{i},dataCut(i,1));
%     scores_learning(i)=sum(score(:,2))/size(score,1);
%     %scores{i}=sum(score(:,2))/size(score,1);
%     %clc;
% end

% if  ~loadMem || ~exist('dataTest','var') 
%     fid = fopen(testList, 'rt');
%     C = textscan(fid, '%s %q %s');
%     fclose(fid);
%     labels = C{3};
%     %что тут такое?
%     [model_ids, ~, Kmodel] = unique(C{1}, 'stable'); % check if the order is the same as above!
%     [test_files, ~, Ktest] = unique(C{2}, 'stable');
%     test_files = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
%                        test_files, 'UniformOutput', false);
%     trials = [Kmodel, Ktest];
% 
%     %% сначала загружаем со всеми признаками
%     %%featAll = 1:featMax;
%     nfiles = length(test_files);
%     dataTest = cell(nfiles, 1);
%     
%     for ix = 1 : nfiles,
%          dataTest{ix} = htkread(test_files{ix});
%     end
% 
%     
% end

mass = cell(2,1);
count = 1;
for features = 1:20
    for j = 1:800
        mass{1}(count) = features;
        mass{2}(count) = j;
        count = count + 1;
    end
end
mass{1} = mass{1}';
mass{2} = mass{2}';
trials = [mass{1}, mass{2}];

test_files = cell(2500,1);
path = 'D:\study\nir\signs\';
count = 1;
for i = 1:50
    for j = 1:50
        test_files{count} = path + string(i) + " (" + string(j) + ").htk";
        count = count + 1;
    end
end

scores = score_gmm_trials(gmm_models, test_files, trials, ubm);


