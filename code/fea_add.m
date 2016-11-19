%% Greedy Add alghorithm for feature selection
featSize = featSize+1;
featCol = indexes(1:featSize);
    
    for j = 1:featMax
    if  ~isempty(find(featCol==j,1)) 
        continue; 
    end;
    
    fprintf('Step ¹ %i, %i \n',i,j);
    featCol(1,featSize) = j;
    
    %%run trial

    core_gmm_ubm; %% eliminate all ubm-gmm time savers, which are not recalculated

    
    %%save results
    oneFeatError(i,j) = eer;
    oneFeatDcf(i,j) = dcf1;
    
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
    
    %% restoring indexes
    indexes(featSize) = I;
    bestFeat(i) = I;
    