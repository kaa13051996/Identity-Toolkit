function [mass_norm, mass_max_min] = normalization(features)
    %нормализация только для 2 колонок признаков
    %features = [500,1;20,30;-10,-9];
    count_features = size(features, 1);
    count_classifiers = size(features, 2);
    mass_max_min = zeros(count_classifiers, 2);
    mass_norm = zeros(count_features, count_classifiers);
    
%     mass_max_min(1, :) = [max(features(:, 1)), min(features(:, 1))]; %для SVM
%     mass_max_min(2, :) = [max(features(:, 2)), min(features(:, 2))]; %для GMM

    for column = 1:count_classifiers
        mass_max_min(column, :) = [max(features(:, column)), min(features(:, column))];
        for str = 1:count_features
            mass_norm(str, column) = (features(str, column) - mass_max_min(column, 2))/ mass_max_min(column, 1);
        end
    end
    
    
%     for i = 1:size_features
%         %mass_max_min(i, :) = [max(features(i, :)), mean(features(i, :))];
%         mass_max_min(i, :) = [max(features(:, i)), min(features(:, i))];
%         for j = 1:2
%             mass_norm(i, j) = (features(i, j) - mass_max_min(i, 2))/ mass_max_min(i, 1);
%         end
%     end        
end