function gmm = gmm_em_gpu_mem_test(dataList, nmix, final_niter, ds_factor, nworkers, gmmFilename,featCol,vadCol,vadThr)
% GPU version of gmm EM alghorithm, basically used for UBM training
% This function uses single thread gpu calculations
% fits a nmix-component Gaussian mixture model (GMM) to data in dataList
% using niter EM iterations per binary split. The process can be
% parallelized in nworkers batches using parfor.
%
% Inputs:
%   - dataList    : ASCII file containing feature file names (1 file per line) 
%					or a cell array containing features (nDim x nFrames). 
%					Feature files must be in uncompressed HTK format.
%   - nmix        : number of Gaussian components (must be a power of 2)
%   - final_iter  : number of EM iterations in the final split
%   - ds_factor   : feature sub-sampling factor (every ds_factor frame)
%   - nworkers    : number of parallel workers
%   - gmmFilename : output GMM file name (optional)
%
% Outputs:
%   - gmm		  : a structure containing the GMM hyperparameters
%					(gmm.mu: means, gmm.sigma: covariances, gmm.w: weights)
%
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center
Console_output = 0; %% Do you need to write training messages?


if ischar(nmix), nmix = str2double(nmix); end
if ischar(final_niter), final_niter = str2double(final_niter); end
if ischar(ds_factor), ds_factor = str2double(ds_factor); end
if ischar(nworkers), nworkers = str2double(nworkers); end

[ispow2, ~] = log2(nmix);
if ( ispow2 ~= 0.5 ),
	error('oh dear! nmix should be a power of two!');
end

if ischar(dataList) || iscellstr(dataList),
	dataList = load_data(dataList,featCol,vadCol,vadThr);
end
if ~iscell(dataList),
	error('Oops! dataList should be a cell array!');
end



nfiles = length(dataList);


if Console_output 
    fprintf('\n\nInitializing the GMM hyperparameters ...\n');
end

globalT = tic;
%нужны данные и в оперативной и в видео пам€ти
dataL = cat(2,dataList{:})';
dataL2 = dataL .* dataL;
datagL = gpuArray(dataL);
datagL2 = gpuArray(dataL2);

partDiv = 100000;
nparts = fix(size(dataL,1) / partDiv) ;


g = gpuDevice;
%dataList = gpuArray(dataList); % transfer data to gpu
%dataL = cell(nfiles, 1); % создаем список с массивами данных на gpu
%for ix = 1 : nfiles
%    dataL{ix} = gpuArray(dataList{ix}(:, 1:ds_factor:end));
%end

% horzcat(dataList{1},dataList{2})
% 1-й вариант - оставл€ем как есть разделение по файлам, батчами делаем
% обучение
% 2-й вариант - объедин€ем все данные в 1 массив

% пока оставлю тут вычислени€ на cpu
[gm, gv] = comp_gm_gv(dataList);
gmm = gmm_init(gm, gv); 

% gradually increase the number of iterations per binary split
% mix = [1 2 4 8 16 32 64 128 256 512 1024];
niter = [1 2 4 4  4  4  6  6   10  10  15];
niter(log2(nmix) + 1) = final_niter;

mix = 1;
while ( mix <= nmix )
	if ( mix >= nmix/2 ), ds_factor = 1; end % not for the last two splits!
    if Console_output 
    fprintf('\nRe-estimating the GMM hyperparameters for %d components ...\n', mix);
    end
    for iter = 1 : niter(log2(mix) + 1)
        if Console_output
        fprintf('EM iter#: %d \t', iter);
        end
        N = 0; F = 0; S = 0; L = 0; nframes = 0;
        tim = tic;
        %gMu = gpuArray(gmm.mu); gSigma = gpuArray(gmm.sigma); gW = gpuArray(gmm.w);
        if mix>32 
            partDiv = fix(100000/(mix/32));
            nparts = fix(size(dataL,1) / partDiv);
            cpuP = fix(nparts/2);
        else
            nparts = 1;
        end
        if (nparts ~= 1)
            p = gcp();
            %p1 = parpool('local',1);
            %run on GPU first
            %[n1, f1, s1, l1] = gpuCycle(nparts-cpuP,partDiv,datagL((cpuP)*partDiv+1:end,:),datagL2((cpuP)*partDiv+1:end,:), gmm);
            
            
            %ftch(cpuP+1) = parfeval(p,@gpuCycle,4,nparts-cpuP,partDiv,datagL((cpuP)*partDiv+1:end,:),datagL2((cpuP)*partDiv+1:end,:), gmm); 
            gpu_basic = tic;
            ftch = parfeval(p,@gpuCycle,4,nparts-cpuP,partDiv,datagL((cpuP)*partDiv+1:end,:),datagL2((cpuP)*partDiv+1:end,:), gmm); 
            [~,n, f, s, l] = fetchNext(ftch);
            gpu_basic = toc(gpu_basic);
            
            cellM = cell(nparts-cpuP,1);
            cellM2 = cell(nparts-cpuP,1);
            cellmu = cell(nparts-cpuP,1);
            cellsigma = cell(nparts-cpuP,1);
            cellw = cell(nparts-cpuP,1);
            

for ix = 1 : nparts-cpuP
	if (ix ~= nparts-cpuP)
	cellM{ix} = datagL((ix-1)*partDiv+1+cpuP*partDiv:ix*partDiv+cpuP*partDiv,:);
	cellM2{ix} = datagL2((ix-1)*partDiv+1+cpuP*partDiv:ix*partDiv+cpuP*partDiv,:);
	else
	cellM{ix} = datagL((ix-1)*partDiv+1+cpuP*partDiv:end,:);
	cellM2{ix} = datagL2((ix-1)*partDiv+1+cpuP*partDiv:end,:);
	end;
    cellmu{ix} = gmm.mu;
    cellsigma{ix} = gmm.sigma;
    cellw{ix} =  gmm.w;
end;

            gpu_arr = tic;
            [n, f, s, ~] = cellfun(@expectation2,cellM, cellM2, cellmu,cellsigma,cellw,'UniformOutput',false);
            gpu_arr = toc(gpu_arr);
            gatherT = tic;
            for ix = 2 : nparts-cpuP
                n{1} = n{1} + n{ix};
                f{1} = f{1} + f{ix};
                s{1} = s{1} + s{ix};
            end
            N = gather(n{1}); F = gather(f{1}); S = gather(s{1});
            gatherT = toc(gatherT);
            
            %“ест с запуском cellFun в отдельном потоке.
             gpu_arr3 = tic;
             ftch = parfeval(p,@gpuCycleCell,3,cellM, cellM2, cellmu,cellsigma,cellw); 
             [~,n, f, s] = fetchNext(ftch);
             for ix = 2 : nparts-cpuP
                n{1} = n{1} + n{ix};
                f{1} = f{1} + f{ix};
                s{1} = s{1} + s{ix};
            end
            N = gather(n{1}); F = gather(f{1}); S = gather(s{1});
             gpu_arr3 = toc(gpu_arr3);
             
            %тест с копированием из RAM
            
            for ix = 1 : nparts-cpuP
	if (ix ~= nparts-cpuP)
	cellM{ix} = dataL((ix-1)*partDiv+1+cpuP*partDiv:ix*partDiv+cpuP*partDiv,:);
	cellM2{ix} = dataL2((ix-1)*partDiv+1+cpuP*partDiv:ix*partDiv+cpuP*partDiv,:);
	else
	cellM{ix} = dataL((ix-1)*partDiv+1+cpuP*partDiv:end,:);
	cellM2{ix} = dataL2((ix-1)*partDiv+1+cpuP*partDiv:end,:);
	end;
    cellmu{ix} = gmm.mu;
    cellsigma{ix} = gmm.sigma;
    cellw{ix} =  gmm.w;
end;

            gpu_arrRAM = tic;
            [n, f, s, ~] = cellfun(@expectation3,cellM, cellM2, cellmu,cellsigma,cellw,'UniformOutput',false);
            gpu_arrRAM = toc(gpu_arrRAM);
            gatherT = tic;
            for ix = 2 : nparts-cpuP
                n{1} = n{1} + n{ix};
                f{1} = f{1} + f{ix};
                s{1} = s{1} + s{ix};
            end
            N = gather(n{1}); F = gather(f{1}); S = gather(s{1});
            gatherT = toc(gatherT);

            %тест с RAM в отдельном потоке.
             gpu_arr4 = tic;
             ftch = parfeval(p,@gpuCycleCellRAM,3,cellM, cellM2, cellmu,cellsigma,cellw); 
             [~,n, f, s] = fetchNext(ftch);
             for ix = 2 : nparts-cpuP
                n{1} = n{1} + n{ix};
                f{1} = f{1} + f{ix};
                s{1} = s{1} + s{ix};
            end
            N = gather(n{1}); F = gather(f{1}); S = gather(s{1});
             gpu_arr4 = toc(gpu_arr4);
            %попробуем параметры гаус смеси закинуть на GPU.
            gmu = gpuArray(gmm.mu);
            gsigma = gpuArray(gmm.sigma);
            gw = gpuArray(gmm.w);
            
            for ix = 1 : nparts-cpuP
                cellmu{ix} = gmu;
                cellsigma{ix} = gsigma;
                cellw{ix} =  gw;
            end;
            gpu_arr2 = tic;
            [n, f, s, ~] = cellfun(@expectation2,cellM, cellM2, cellmu,cellsigma,cellw,'UniformOutput',false);
            gpu_arr2 = toc(gpu_arr2);
            
            
            
            
            %попробуем закинуть сюда parfor
            %parfor (ix = 1 : cpuP, nworkers-1)
            %    [n, f, s, l] = expectation2(dataL((ix-1)*partDiv+1:ix*partDiv,:),dataL2((ix-1)*partDiv+1:ix*partDiv,:), gmm.mu,gmm.sigma,gmm.w);
            %    N = N + n; F = F + f; S = S + s; L = L + sum(l);
            %end;
            %for idx = 1:cpuP
            %    ftch(idx) = parfeval(p,@expectation2,4,dataL((idx-1)*partDiv+1:idx*partDiv,:),dataL2((idx-1)*partDiv+1:idx*partDiv,:), gmm.mu,gmm.sigma,gmm.w); % Square size determined by idx
            %end
            
            %only cpu test
            %for idx = 1:nparts
            %    if (idx ~= nparts)
            %        ftch(idx) = parfeval(p,@expectation2,4,dataL((idx-1)*partDiv+1:idx*partDiv,:),dataL2((idx-1)*partDiv+1:idx*partDiv,:), gmm.mu,gmm.sigma,gmm.w); % Square size determined by idx
            %    else
            %        ftch(idx) = parfeval(p,@expectation2,4,dataL((idx-1)*partDiv+1:end,:),dataL2((idx-1)*partDiv+1:end,:), gmm.mu,gmm.sigma,gmm.w); % Square size determined by idx
            %    end;
            %end
            
            % Collect the results as they become available.
            
            %for idx = 1:cpuP+1
            %for idx = 1:nparts
            
            % fetchNext blocks until next results are available.
                [~,n, f, s, l] = fetchNext(ftch);
                N = N + n; F = F + f; S = S + s; L = L + sum(l);
               
            %end
            
        else
            [n, f, s, l] = expectation2(datagL,datagL2,gmm.mu,gmm.sigma,gmm.w); %, gMu,gSigma,gW);
           
            N = gather(n); F = gather(f); S = gather(s); %L =  gather(sum(l));
			%nframes = nframes + length(l);
        end 
        
        tim = toc(tim);
        if Console_output 
        fprintf('[llk = %.2f] \t [elaps = %.2f s]\n', L/nframes, tim);
        end
        
        %test maximization time using gpu and cpu
        %maxT = tic;
        gmm = maximization(N,F,S);
        %gmm = maximization(N, F, S);
        %gmm.mu = gather(gmm.mu);
        %gmm.w = gather(gmm.w);
        %gmm.sigma = gather(gmm.sigma);
        %maxT = toc(maxT);
        %disp(maxT);
        %disp(g.FreeMemory);
        %wait(g);
    end
    if ( mix < nmix ), 
        gmm = gmm_mixup(gmm); 
    end
    mix = mix * 2;
end

globalT = toc(globalT); 
disp (globalT);
if ( exist('gmmFilename','var') && ~isempty(gmmFilename)),
	fprintf('\nSaving GMM to file %s\n', gmmFilename);
	% create the path if it does not exist and save the file
	path = fileparts(gmmFilename);
	if ( exist(path, 'dir')~=7 && ~isempty(path) ), mkdir(path); end
	save(gmmFilename, 'gmm');
end

function [N, F, S, L] = gpuCycle(nparts,partDiv,dataL,dataL2,gmm)
N = zeros(1,'gpuArray');
F = zeros(1,'gpuArray');
S = zeros(1,'gpuArray');
L = zeros(1,'gpuArray');
for ix = 1 : nparts % “ут надо поэкспериментировать с количеством workers, либо убрать в принципе parfor
%         for ix = 1 : nfiles,
            %[n, f, s, l] = expectation(dataL{ix}, gmm);
            
            if (ix ~= nparts)
            [n, f, s, l] = expectation2(dataL((ix-1)*partDiv+1:ix*partDiv,:),dataL2((ix-1)*partDiv+1:ix*partDiv,:), gmm.mu,gmm.sigma,gmm.w);
            else
                [n, f, s, l] = expectation2(dataL((ix-1)*partDiv+1:end,:),dataL2((ix-1)*partDiv+1:end,:),gmm.mu,gmm.sigma,gmm.w); %, gMu,gSigma,gW);
            end
            N = N + n; F = F + f; S = S + s; L = L + sum(l);
			%nframes = nframes + length(l);
end
N = gather(N); F = gather(F); S = gather(S); L = gather(L);

function [N, F, S] = gpuCycleCell(dataL,dataL2,mu,sigma,w)
%N = zeros(1,'gpuArray');
%F = zeros(1,'gpuArray');
%S = zeros(1,'gpuArray');
%L = zeros(1,'gpuArray');

[N, F, S,~] = cellfun(@expectation2,dataL, dataL2, mu,sigma,w,'UniformOutput',false);
%N = gather(N); F = gather(F); S = gather(S); %L = gather(L);

function [N, F, S] = gpuCycleCellRAM(dataL,dataL2,mu,sigma,w)

[N, F, S,~] = cellfun(@expectation3,dataL, dataL2, mu,sigma,w,'UniformOutput',false);

function data = load_data(datalist,featCol,vadCol,vadThr)
% load all data into memory
if ~iscellstr(datalist)
    fid = fopen(datalist, 'rt');
    filenames = textscan(fid, '%q');
    fclose(fid);
    filenames = filenames{1};
else
    filenames = datalist;
end
nfiles = size(filenames, 1);
data = cell(nfiles, 1);
if nargin == 2
for ix = 1 : nfiles,
    data{ix} = htkread(filenames{ix},featCol);
end
else
        if nargin == 1
    for ix = 1 : nfiles,
    data{ix} = htkread(filenames{ix});
    end
        else % all parameters
            if ~isempty(vadCol)
                for ix = 1 : nfiles,
                data{ix} = htkread(filenames{ix},featCol,vadCol,vadThr);
                end
            else
                for ix = 1 : nfiles,
                data{ix} = htkread(filenames{ix},featCol);
                end    
            end
        end
end

function [gm, gv] = comp_gm_gv(data)
% computes the global mean and variance of data
nframes = cellfun(@(x) size(x, 2), data, 'UniformOutput', false);
nframes = sum(cell2mat(nframes));
gm = cellfun(@(x) sum(x, 2), data, 'UniformOutput', false);
gm = sum(cell2mat(gm'), 2)/nframes;
gv = cellfun(@(x) sum(bsxfun(@minus, x, gm).^2, 2), data, 'UniformOutput', false);
gv = sum(cell2mat(gv'), 2)/( nframes - 1 );

function gmm = gmm_init(glob_mu, glob_sigma)
% initialize the GMM hyperparameters (Mu, Sigma, and W)
gmm.mu    = glob_mu;
gmm.sigma = glob_sigma;
gmm.w     = 1;

function [N, F, S, llk] = expectation3(data, data2, mu, sigma, w)
% compute the sufficient statistics
data = gpuArray(data);
data2 = gpuArray(data2);

muDivSig = mu./sigma;

ndim = size(data, 2);
C = sum(mu.*muDivSig) + sum(log(sigma));
D = data2 * (1./sigma)  - 2 * data * (muDivSig) + ndim * log(2 * pi);

post = -0.5 * (bsxfun(@plus, C,  D));
clear C D;

post = bsxfun(@plus, post, log(w));

xmax = max(post, [], 2);
llk    = xmax + log(sum(exp(bsxfun(@minus, post, xmax)), 2));
ind  = find(~isfinite(xmax));
if ~isempty(ind)
    llk(ind) = xmax(ind);
end

post = exp(bsxfun(@minus, post, llk));

N = sum(post, 1);
F = (post' * data)';
S = (post' * data2)';


function [N, F, S, llk] = expectation2(data, data2, mu, sigma, w)
% compute the sufficient statistics

%post = lgmmprob(data,gmm.mu, gmm.sigma,gmm.w(:));
muDivSig = mu./sigma;
%oneDivSigDat = (1./gmm.sigma)'* data2;
ndim = size(data, 2);
C = sum(mu.*muDivSig) + sum(log(sigma));
D = data2 * (1./sigma)  - 2 * data * (muDivSig) + ndim * log(2 * pi);
%D = oneDivSigDat - 2 * (muDivSig)' * data + ndim * log(2 * pi);
post = -0.5 * (bsxfun(@plus, C,  D));
clear C D;
%wait(gpuDevice);
post = bsxfun(@plus, post, log(w));

%llk  = logsumexp(post, 1);
xmax = max(post, [], 2);
llk    = xmax + log(sum(exp(bsxfun(@minus, post, xmax)), 2));
ind  = find(~isfinite(xmax));
if ~isempty(ind)
    llk(ind) = xmax(ind);
end

post = exp(bsxfun(@minus, post, llk));

N = sum(post, 1);
F = (post' * data)';
S = (post' * data2)';

function [N, F, S, llk] = expectation(data, gmm)
% compute the sufficient statistics
[post, llk] = postprob(data, gmm.mu, gmm.sigma, gmm.w(:));
N = sum(post, 2)';
F = data * post';
S = (data .* data) * post';

function [post, llk] = postprob(data, mu, sigma, w)
% compute the posterior probability of mixtures for each frame
post = lgmmprob(data, mu, sigma, w);
llk  = logsumexp(post, 1);
post = exp(bsxfun(@minus, post, llk));

function logprob = lgmmprob(data, mu, sigma, w)
% compute the log probability of observations given the GMM
ndim = size(data, 1);
C = sum(mu.*mu./sigma) + sum(log(sigma));
D = (1./sigma)' * (data .* data) - 2 * (mu./sigma)' * data  + ndim * log(2 * pi);
logprob = -0.5 * (bsxfun(@plus, C',  D));
logprob = bsxfun(@plus, logprob, log(w));

function y = logsumexp(x, dim)
% compute log(sum(exp(x),dim)) while avoiding numerical underflow
xmax = max(x, [], dim);
y    = xmax + log(sum(exp(bsxfun(@minus, x, xmax)), dim));
ind  = find(~isfinite(xmax));
if ~isempty(ind)
    y(ind) = xmax(ind);
end

function gmm = maximization(N, F, S)
% ML re-estimation of GMM hyperparameters which are updated from accumulators
w  = N / sum(N);
mu = bsxfun(@rdivide, F, N);
sigma = bsxfun(@rdivide, S, N) - (mu .* mu);
sigma = apply_var_floors(w, sigma, 0.1);
gmm.w = w;
gmm.mu= mu;
gmm.sigma = sigma;

function sigma = apply_var_floors(w, sigma, floor_const)
% set a floor on covariances based on a weighted average of component
% variances
vFloor = sigma * w' * floor_const;
sigma  = bsxfun(@max, sigma, vFloor);
% sigma = bsxfun(@plus, sigma, 1e-6 * ones(size(sigma, 1), 1));

function gmm = gmm_mixup(gmm)
% perform a binary split of the GMM hyperparameters
mu = gmm.mu; sigma = gmm.sigma; w = gmm.w;
[ndim, nmix] = size(sigma);

if ndim == 1
    sig_max = sigma;
    arg_max = ones(1,nmix);
else 
    [sig_max, arg_max] = max(sigma);
end;

eps = sparse(0 * mu);
eps(sub2ind([ndim, nmix], arg_max, 1 : nmix)) = sqrt(sig_max);
% only perturb means associated with the max std along each dim 
mu = [mu - eps, mu + eps];
% mu = [mu - 0.2 * eps, mu + 0.2 * eps]; % HTK style
sigma = [sigma, sigma];
w = [w, w] * 0.5;
gmm.w  = w;
gmm.mu = mu;
gmm.sigma = sigma;
