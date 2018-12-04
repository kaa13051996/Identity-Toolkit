function [features] = get_features(scores_file)
%     scores = ['D:\study\nir\Identity-Toolkit\code\scores_svm'; 'D:\study\nir\Identity-Toolkit\code\scores-2'];
%     features = cell(2,1);
%     for file = 1 : size(scores, 1)
%         if isfile(scores(file))
%             fid = fopen(scores(file), 'r');
%             features{file} = fread(fid, 'double');
%             fclose(fid);
%         else
%             fprintf('Файла не существует!\n');
%         end
%     end
    features = zeros(1,1);
    if isfile(scores_file)
        fid = fopen(scores_file, 'r');
        features = fread(fid, 'double');
        fclose(fid);
    else
        fprintf('Файла не существует!\n');
    end
end