clc
clear

%% Step0: Opening MATLAB pool
nworkers = 8;
nworkers = min(nworkers, feature('NumCores'));

p = gcp('nocreate'); 
if isempty(p), parpool(nworkers); end 

%% Loading config file
fea_dir =  'D:\study\nir\signs\'; % Feature files list
%fea_dir =  'D:\study\nir\Smile_full\'; % Feature files list
configDir = 'D:\study\nir\data\ListsMF\';
ini = IniConfig();
%ini.ReadFile(strcat(configDir,'cDBN3.lst'));
ini.ReadFile(strcat(configDir,'cAddDel1.lst'));
sections = ini.GetSections();
indGMM = find(ismember(sections,'[UBM GMM]'));
indDS = find(ismember(sections,'[Data selection]'));
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

