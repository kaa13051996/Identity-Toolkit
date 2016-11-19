function  fea_show_tables( steps, Err, Dcf, idx )
%FEA_SHOW_TABLES Summary of this function goes here
%   Detailed explanation goes here
fprintf('%6s %6s %3s %6s \n','Step ¹', 'idx','EER','Dcf');
for st=1:steps
    fprintf('%6i %3i %6.3f %6.3f \n',st,idx(st),Err(st,idx(st)),Dcf(st,idx(st)));
end

end

