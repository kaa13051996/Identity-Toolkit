function [scores] = get_scores(models, data, filename)
    scores = zeros(size(data,1)*size(models,1),1);        
    for i=1:size(models,1)     
        for j=1:size(data,1)
            [~,score] = predict(models{i},data(j,:));
            scores(((i-1)*size(data,1))+j)=score(2);%sum(score(:,2))/size(score,1);
        %scores{i}=sum(score(:,2))/size(score,1);
        %clc;
    end    
    if isfile(filename)
        fid = fopen(filename,'w');
        fwrite(fid, scores, 'double');
        fclose(fid);
    else
         fprintf('Файла не существует!\n');
    end    
end