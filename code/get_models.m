function [svm, mass, mass_labels] = get_models(dataCut, nspks)
    models = cell(nspks, 1);
    count_speakers = nspks;%20
    count_recordings = length(dataCut{nspks});%40
    count_features = length(dataCut{nspks}{1});%28
    mass = zeros(count_speakers*count_recordings, count_features);%dataCut в виде таблицы
    var = 1;

    for dictor = 1 : count_speakers
        for audio = 1 : count_recordings
            for feature = 1 : count_features
                mass(var, feature) = dataCut{dictor}{audio}(feature);            
            end
            var = var + 1;
        end
    end
    
    mass_labels = cell(count_speakers,1);
    
    for spk = 1 : nspks
        labels = repmat(0, [count_speakers*count_recordings,1]); %создаем массив 0 для spk-модели
        fprintf('.');
        %set labels for speaker
        labels(spk*count_recordings+1-count_recordings:spk*count_recordings,1) = 1; %заполняем 1 там, где записи spk-диктора    
        mass_labels{spk} = labels;
        labels=int2str(labels(:));        
        models{spk} = fitcsvm(mass,labels,'KernelFunction','linear', 'Standardize', true );
%         models{spk} = fitcsvm(mass,labels,'KernelFunction','gaussian', 'Standardize', true );
%         models{spk} = fitcsvm(mass,labels,'KernelFunction','rbf', 'Standardize', true );
%         models{spk} = fitcsvm(mass,labels,'KernelFunction','polynomial', 'Standardize', true );
%         models{spk} = fitcdiscr(mass,labels,'DiscrimType','pseudoLinear'); %IDA
%         models{spk} = fitcnb(mass,labels); %bayes -
%         models{spk} = fitensemble(mass,labels,'AdaBoostM1',100,'Tree');
%         %ensemble
%         models{spk} = TreeBagger(50,mass,labels ); %TreeBagger
    end
    svm = models;
end