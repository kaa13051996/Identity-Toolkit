
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
C = textscan(fid, '%s %q');%��������� ��� ������ ����� �� ��������
fclose(fid);
model_ids = unique(C{1}, 'stable'); %���������� �� ��������������� �������� C[1]
model_files = C{2};
nspks = length(model_ids);
gmm_models = cell(nspks, 1); %���������� ������ 20*1
dataTrain = cell(nspks,1); %��� ��� ����������??

for spk = 1 : nspks,
    ids = find(ismember(C{1}, model_ids{spk})); %����� � model_ids[i] �������� �� �[i], ������� �� �������
    spk_files = model_files(ids); %���� ������ ��� �����
    spk_files = cellfun(@(x) fullfile(fea_dir, x),...  %# ���� ������ ���� �� �����
                       spk_files, 'UniformOutput', false);
    %%��������� ��������� ������
    dataTrain{spk} = load_data_svm(spk_files);
end
end

%%�������� �������� ����
dataCut = dataTrain;
%for spk = 1 : nspks, %�� 1 �� 20
%   for spk2 = 1: length(dataTrain{spk}) %�� 1 �� 40
%       dataCut{spk}{spk2} = dataCut{spk}{spk2}(featCol,:);
%   end
%end
%%��� ��� � ������������ ������� �������� ���������

[models, data, labels] = get_models(dataCut, nspks); %���������� ������, ������ ��������� � �����
filename = 'D:\study\nir\Identity-Toolkit\code\scores_svm';
scores_learning = get_scores(models, data, filename);

