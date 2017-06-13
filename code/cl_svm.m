rng(1);
%%Load data and set labels
    load('E:\temp\123\data\DBN\sp1.mat');
    %%
    dataCut=out1;
    nspks = length(dataCut);
    models = cell(nspks, 1); 
    %concat speakers
    sp_files = length(dataCut{1});
    count = 0;

    %dataCut = dataCut{:};
    for i=1:nspks
        for j=1:sp_files
            count = count + size(dataCut{i}{j},2);
        end
    end

    %concat data
numfeat = size(dataCut{1}{1},1);
Pconc = zeros( count,numfeat); 
startframe_list = zeros(nspks+1,1);

count = 0;
for i=1:nspks
    startframe_list(i) = count+1;
        for j=1:sp_files
             Pconc( count+1:count+size(dataCut{i}{j},2),:) = dataCut{i}{j}';
            count = count + size(dataCut{i}{j},2);
        end
end
startframe_list(nspks+1) = size(Pconc,1)+1;

dataCut = Pconc;

%create labels
%%
    for spk = 1 : nspks,
        fprintf('.');
        %set labels for speaker
        labels = zeros(size(dataCut,1),1);
        labels(startframe_list(spk):startframe_list(spk+1)-1,1) = ones(startframe_list(spk+1)-startframe_list(spk),1);
        labels=int2str(labels(:));
        %train model
        models{spk} = fitcsvm(dataCut,labels,'KernelFunction','linear' );
        %models{spk} = fitSVMPosterior(models{spk});
        %models{spk} = fitcdiscr(dataCut,labels,'DiscrimType','pseudoLinear');
        %ldaClass = resubPredict(lda);
    %models{spk} = mapAdapt(dataCut{spk}, ubm, map_tau, config,'',featCol,vadCol,vadThr);
    end
disp('\n');
%% load test data and test it

    load('E:\temp\123\data\DBN\tst1.mat');
    load('E:\temp\123\data\DBN\labels.mat');
    %%
    dataCut=out1';
    for i=1:length(dataCut)
        dataCut{i}=dataCut{i}';
    end
    scores = zeros(size(trials,1),1);
    for i=1:size(trials,1)
        
        [~,score] = predict(models{trials(i,1)},dataCut{trials(i,2)});
        scores(i)=sum(score(:,2))/size(score,1);
    end
%% compute eer
[eer, dcf1, dcf2] = compute_eer(scores, labels, true);

%%
%thr=0.25;
%rgt1 = 0;
%rgt2 = 0;
%for i=1:size(scores,1)
%   if (scores(i)>=thr)&&labels(i)==1
%       rgt1=rgt1+1;
%   end
%   if (scores(i)<thr)&&labels(i)==0
%       rgt2=rgt2+1;
%   end
%end
%err = size(scores,1)-rgt1-rgt2