function [] = write_scores(scores, filename)
    if isfile(filename)
        fid = fopen(filename,'w');
        fwrite(fid, scores, 'double');
        fclose(fid);
    else
         fprintf('����� �� ����������!\n');
    end    
end