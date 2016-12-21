%%алгортим жадного удаления
%%!!!
%% не забываем поменять i, чтобы результаты записались куда надо!
%%!!!
    %%featSize = size(featCol,2); %% устанавливаем нужную длину(+1 для Add, = для Del)

    featCol = indexes(1:featSize); %%поменять i
    fprintf('Greedy Del. Step № %i\n',i);

    for j = 1:featSize
    
    
    featCol = indexes(1:featSize); %% restore feature vector
    featInd = featCol(j); %% writing down feature number
    featCol(j) = []; %% deleting j-th feature


    
    %%run trial

    core_gmm_ubm; %% eliminate all ubm-gmm time savers, which are not recalculated

    
    %%save results
    oneFeatError(i,featInd) = eer;
    oneFeatDcf(i,featInd) = dcf1;
    
    end
    %%finding best result
    
    [M,I] = min(oneFeatError(i,:));
    
    
    if I~=featMax
    cur = I;
    for k = I+1:featMax
        if (oneFeatError(i,k) == M) && (oneFeatDcf(i,cur)>oneFeatDcf(i,k))
            cur = k;
        end;
    end;
    I=cur;

    end;
    %%indexes(i) = I; delete I-th feature from vector
    indexes(find(indexes==I)) = indexes(featSize); 
    indexes(featSize) = 0;
    
    bestFeat(i) = I;
    featSize = featSize -1;