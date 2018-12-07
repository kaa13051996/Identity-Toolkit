function [mass_norm] = normalization_test(features, param)
    count_features = size(features, 1);
    count_classifiers = size(features, 2);
    mass_norm = zeros(count_features, count_classifiers);
    for column = 1:count_classifiers        
        for str = 1:count_features
            mass_norm(str, column) = (features(str, column) - param(column, 2))/ param(column, 1);
        end
    end
        
end