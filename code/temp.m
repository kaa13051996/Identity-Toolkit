for i = 46 : 46
    tic;
    if method == 0 
        fea_add;
    else
        fea_del;
    end
    toc
    
       %%output result
       if (i > 1)
            fprintf('Result of previous step, EER= %6.3f, dcf= %6.3f \n',oneFeatError(i-1,bestFeat(i-1)),oneFeatDcf(i-1,bestFeat(i-1)));
       end
       fprintf('Result of  step № %i, EER= %6.3f, dcf= %6.3f \n',i,M,oneFeatDcf(i,I));

    if (i == 1) || (oneFeatError(i,I)>oneFeatError(i-1,bestFeat(i-1))) % если у нас ошибка не уменьшилась, то увеличиваем счетчик
       current_steps = current_steps + 1;
    else
        if (oneFeatError(i,I)<oneFeatError(i-1,bestFeat(i-1)))
        current_steps = 0;
        end
    end
    
    if (current_steps == method_ch_steps+1 && method == 1) % если сделали все, что смогли, заканчиваем
        break;
    end
    
    if (current_steps == method_ch_steps && method ~= 1)
        current_steps = 0;
        method = 1;
    end
      
    
end