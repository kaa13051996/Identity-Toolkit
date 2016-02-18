function [data, frate, feakind] = htkread(filename,ind)
% reads features with HTK format
% ind - row of column's indexes to be read
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

%fid = fopen(filename, 'rb', 'ieee-be');
fid = fopen(filename, 'rb', 'native');
nframes = fread(fid, 1, 'int32'); % number of frames
frate   = fread(fid, 1, 'int32'); % frame rate in 100 nano-seconds unit
nbytes  = fread(fid, 1, 'short'); % number of bytes per feature value
feakind = fread(fid, 1, 'short'); % 9 is USER
ndim = nbytes / 4; % feature dimension (4 bytes per value)
data = fread(fid, [ndim, nframes], 'float');
fclose(fid);
if ( nargin == 2 && ndim > size(ind,2))    
data = data(ind,:);
end