function [features] = get_features(scores, count_models)
    scores = ['D:\study\nir\Identity-Toolkit\code\scores-1'; 'D:\study\nir\Identity-Toolkit\code\scores-2'];
    features = cell(2,1);
    for file = 1 : size(scores, 1)
        if isfile(scores(file))
            fid = fopen(scores(file), 'r');
            features{file} = fread(fid, 'double');
            fclose(fid);
        else
            fprintf('Файла не существует!\n');
        end
    end
end