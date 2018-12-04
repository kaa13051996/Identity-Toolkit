
% %% Step1: Training the UBM 
loadMem = true; %% load all files into the memory
loadUBM = false; %% load UBM from disk

load(ubmFile);
ubm = gmm;
clear('gmm');    

%% Step2: Adapting the speaker models from UBM
map_tau = 10.0;
config = 'm';

if  ~loadMem || ~exist('dataTrain','var') 
fid = fopen(trainList, 'rt');
C = textscan(fid, '%s %q');%прочитать как массив ячеек со строками
fclose(fid);
model_ids = unique(C{1}, 'stable'); %уникальные не отсортированные значения C[1]
model_files = C{2};
nspks = length(model_ids);
gmm_models = cell(nspks, 1); %одномерный массив 20*1
dataTrain = cell(nspks,1); %что там содержится??

for spk = 1 : nspks,
    ids = find(ismember(C{1}, model_ids{spk})); %найти в model_ids[i] значения из С[i], которые не нулевые
    spk_files = model_files(ids); %было просто имя файла
    spk_files = cellfun(@(x) fullfile(fea_dir, x),...  %# стал полный путь до файла
                       spk_files, 'UniformOutput', false);
    %%загружаем обучающие данные
    dataTrain{spk} = load_data_svm(spk_files);
end
end

%%вырезаем ненужные фичи
dataCut = dataTrain;
%for spk = 1 : nspks, %от 1 до 20
%   for spk2 = 1: length(dataTrain{spk}) %от 1 до 40
%       dataCut{spk}{spk2} = dataCut{spk}{spk2}(featCol,:);
%   end
%end
%%тут уже с загруженными данными проводим адаптацию

[models, data, labels] = get_models(dataCut, nspks); %возвращает модели, массив признаков и метки
filename = 'D:\study\nir\Identity-Toolkit\code\scores_svm';
scores_learning = get_scores(models, data, filename);

