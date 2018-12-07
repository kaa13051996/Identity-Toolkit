function [mass_labels] = get_labels(count_speakers, count_records_all)    
    %mass_labels = cell(count_speakers,1);
    count_record_one = 40; %у 1 диктора 40 аудиозаписей, всего дикторов 20
    
    for model = 1:1 %20
        labels = zeros(count_speakers * count_records_all,1);
        for var = model*count_record_one+1-count_record_one:count_records_all+count_record_one:count_records_all*count_speakers %20*800=16000
            labels(var:var+count_record_one-1) = 1;
        end
        mass_labels = labels;
    end
end