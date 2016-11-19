%%set needed features in config file
load_cfg;
maxSteps = 40; %% максимум шагов жадного алгоритма

featMax = max(featCol); %% максимальное количество признаков
if maxSteps>featMax 
    maxSteps = featMax;
end;
bestFeat = 1:maxSteps; %% массив признаком, найденным на i-ом шаге (добавленным или удаленным)
indexes = zeros(1,featMax); %% indexes of the best features
featSize = 0; %% current size of feature vector


%%first greedy method
%%adding one feature at a time

oneFeatError = 100*ones(featMax);
oneFeatDcf = 10000*ones(featMax);

for i = 1 : maxSteps,
    fea_add;
    
       %%output result
    fprintf('Result of previous step, EER= %6.3f, dcf= %6.3f \n',oneFeatError(i-1,bestFeat(i-1)),oneFeatDcf(i-1,bestFeat(i-1)));
    fprintf('Result of  step № %i, EER= %6.3f, dcf= %6.3f \n',i,M,oneFeatDcf(i,I));

    
end

%%get and save results


