function [mass_labels] = get_labels(count_speakers, count_records)    
    mass_labels = cell(count_speakers,1);
    count_one = 40;
    
    for model = 1:count_speakers %20
        labels = zeros(count_speakers * count_records,1);
        for var = model*count_one+1-count_one:count_records:count_records*count_speakers %20*800=16000
            labels(var:var+count_one-1) = 1;
        end
        mass_labels{model} = labels;
    end
end