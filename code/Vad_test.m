clear;
fea_dir =  'E:\temp\123\Data\Vad\'; % Feature files list
config_name = 'E:\temp\123\Data\vad.cfg'; %
fid = fopen(config_name, 'rt');
filenames = textscan(fid, '%q %q');
fclose(fid);
filenames = cellfun(@(x) fullfile(fea_dir, x),...  %# Prepend path to files
                       filenames, 'UniformOutput', false);
                   
for i=1:size(filenames{1},1)
    
    true = 0;
    err2 = 0;
    err1 = 0;
    false1 = 0;
    
   data = htkread(char(filenames{1}(i))); 
   vadCol = 20;
   vad_mean = mean(data(vadCol,:));
   A = data(vadCol,:);
   
   vad_min = min(A(A>0));
   vad_thr = vad_mean-(vad_mean-vad_min)/2;
   toDelete = data(vadCol,:)<=vad_thr;
    
   toDelete_manual = false(size(toDelete));
   fid = fopen(char(filenames{2}(i)), 'rt');
    pauses_t = textscan(fid, '%s');
   pauses = double.empty(0);
    for j=1:size(pauses_t{1},1)
        pauses = cat(2,pauses, str2num(char(pauses_t{1}(j))));
    end
    pauses = pauses + 1;
    for j=1:size(pauses,2)
       
       toDelete_manual(pauses(1,j))=1; 
    end
    fclose(fid);
    if size(toDelete_manual,2)>size(toDelete)
      toDelete_manual = toDelete_manual(1,1:size(toDelete,2));
    end
    
    for j=1:size(toDelete_manual,2)
    if (toDelete_manual(1,j) == toDelete(1,j))&& (toDelete(1,j) == 1) 
        true = true + 1; end
    if (toDelete_manual(1,j) == toDelete(1,j))&& (toDelete(1,j) == 0) 
        false1 = false1 + 1; end
    if (toDelete_manual(1,j) ~= toDelete(1,j))&& (toDelete_manual(1,j) == 1) 
        err2 = err2 + 1; end
    if (toDelete_manual(1,j) ~= toDelete(1,j))&& (toDelete_manual(1,j) == 0) 
        err1 = err1 + 1; end
    
    end
    true = true / size(pauses,2); %% верно определенные паузы, %
    false1 = false1 / (j-size(pauses,2)); %% верно определенная речь, %
    err2 = err2 / size(pauses,2); %% определили речь, а там пауза
    %% err1 - определили паузу, а там речь.
    
end