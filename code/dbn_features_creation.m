addpath 'E:\work\Aspirant\CRBM\crbm_audio\voicebox';
addpath 'E:\work\Aspirant\CRBM\crbm_audio\code';

load ('E:\work\Aspirant\CRBM\crbm_audio\results\audio\tirbm_audio_LB_TIMIT_V1b_w6_b300_pc80_sigpc3_p0.05_pl0.05_plambda5_sp3_exp_eps0.01_epsdecay0.01_l2reg0.01_bs01_20170412T094826.mat');
load ('E:\work\Aspirant\CRBM\crbm_audio\results\audio\tirbm_audio_LB2_TIMIT_V1b_w6_b300_pc80_sigpc3_p0.05_pl0.05_plambda5_sp3_exp_eps0.008_epsdecay0.01_l2reg0.02_bs01_20170414T073903_min.mat');
load ('E:\work\Aspirant\CRBM\crbm_audio\results\audio\tirbm_audio_LB3_TIMIT_V1b_w6_b300_pc80_sigpc3_p0.05_pl0.05_plambda5_sp3_exp_eps0.00769231_epsdecay0.01_l2reg0.02_bs01_20170415T111034_0023EPOCHS.mat');

%Make features for UBM
%%
    speakers = [1:15,26:40];
    files = [1:50];
    d = 'E:\temp\123\';
    flist = [];

    for sp=1:length(speakers)
        for file=1:length(files)
            flist{length(flist)+1} = strcat(d,int2str(speakers(sp)),' (',int2str(files(file)),').wav');
        end;
    end;

    Pall = {};
    parfor i= 1:length(flist)
        if mod(i,10)==0, fprintf('.'); end
        if mod(i,1000)==0, fprintf('\n'); end
         % read wav file
        [y,fs,~]=readwav(flist{i});
        % convert to spectrogram
        %Pall{i} = get_spectrogram_orig(y, 0, fs);
        nfft = ceil(fs*0.02);
        WINDOW = hamming(nfft);
        noverlap = nfft-ceil(fs*0.01);

        [Pall{i}, ~] = spectrogram(y, WINDOW, noverlap, nfft, fs);
        Pall{i} = flipud(abs(Pall{i}));
        Pall{i} = log(1e-5+Pall{i}) - log(1e-5);
        Pall{i} = Pall{i} - mean(mean(Pall{i}));
        Pall{i} = Pall{i} / sqrt(mean(mean(Pall{i}.^2)));

    end
    
    ws=pars.ws;
    spacing = pars.spacing;
    out1 = cell(1,length(flist));
    out2 = cell(1,length(flist));
    out3 = cell(1,length(flist));
    
    parfor i= 1:length(flist)
    
    Xw = Ewhiten*Pall{i};
    if mod(size(Xw,2)-ws+1, spacing)~=0
            n = mod(size(Xw,2)-ws+1, spacing);
            Xw(:, 1:floor(n/2), :) = [];
            Xw(:, end-ceil(n/2)+1:end, :) = [];
    end
    Xwtr = Xw';
    Xwtr = reshape(Xwtr, [size(Xwtr,1), 1, size(Xwtr,2)]);

    %получаем выход с 1 слоя
        sends = tirbm_inference_fixconv_1d(Xwtr, W, hbias_vec, pars);
        [~,~,~, hpc] = tirbm_sample_multrand_1d(sends, spacing);
        out1{i}=hpc;
        %обрезаем, чтобы данные соответствовали количеству spacing
        if mod(size(hpc,1)-ws+1, spacing)~=0
            n = mod(size(hpc,1)-ws+1, spacing);
            hpc(1:floor(n/2), :,  :) = [];
            hpc(end-ceil(n/2)+1:end, :,  :) = [];
        end
     %получаем выход с 2 слоя
        sends = tirbm_inference_fixconv_1d(hpc, W2, hbias_vec2, pars);
        [~,~,~, hpc2] = tirbm_sample_multrand_1d(sends, spacing);
        out2{i}=hpc2;
        %обрезаем, чтобы данные соответствовали количеству spacing
        if mod(size(hpc2,1)-ws+1, spacing)~=0
            n = mod(size(hpc2,1)-ws+1, spacing);
            hpc2(1:floor(n/2), :,  :) = [];
            hpc2(end-ceil(n/2)+1:end, :,  :) = [];
        end
        %получаем выход с 3 слоя
        sends = tirbm_inference_fixconv_1d(hpc2, W3, hbias_vec3, pars);
        [~,~,~, out3{i}] = tirbm_sample_multrand_1d(sends, spacing);
        
     
    end;
    
     
    for i=1:length(flist)
        out1{i}=squeeze(out1{i})';
        out2{i}=squeeze(out2{i})';
        out3{i}=squeeze(out3{i})';
    end;
    
    save('E:\temp\123\data\DBN\ubm1.mat','out1');
    save('E:\temp\123\data\DBN\ubm2.mat','out2');
    save('E:\temp\123\data\DBN\ubm3.mat','out3');
	
   %% Make features for speaker models
%
    speakers = [16:25,41:50];
    files = [1:40];
    d = 'E:\temp\123\';
    flist = cell(length(speakers),1);
    
    
    for sp=1:length(speakers)
        flist{sp}=cell(length(files),1);
        for file=1:length(files)
            flist{sp}{file} = strcat(d,int2str(speakers(sp)),' (',int2str(files(file)),').wav');
        end;
    end;

    Pall = cell(length(speakers),1);
    
    parfor sp=1:length(speakers)
        Pall{sp}=cell(length(files),1);
        for i= 1:length(files)
        if mod(i,10)==0, fprintf('.'); end
         % read wav file
        [y,fs,~]=readwav(flist{sp}{i});
        % convert to spectrogram
        %Pall{i} = get_spectrogram_orig(y, 0, fs);
        nfft = ceil(fs*0.02);
        WINDOW = hamming(nfft);
        noverlap = nfft-ceil(fs*0.01);

        [Pall{sp}{i}, ~] = spectrogram(y, WINDOW, noverlap, nfft, fs);
        Pall{sp}{i} = flipud(abs(Pall{sp}{i}));
        Pall{sp}{i} = log(1e-5+Pall{sp}{i}) - log(1e-5);
        Pall{sp}{i} = Pall{sp}{i} - mean(mean(Pall{sp}{i}));
        Pall{sp}{i} = Pall{sp}{i} / sqrt(mean(mean(Pall{sp}{i}.^2)));

        end
    end
    fprintf('\n');
    
    ws=pars.ws;
    spacing = pars.spacing;
    out1 = cell(length(speakers),1);
    out2 = cell(length(speakers),1);
    out3 = cell(length(speakers),1);
    
    parfor sp=1:length(speakers)
        tmp = cell(length(files),1);
        tmp2 = cell(length(files),1);
        tmp3 = cell(length(files),1);
    for i= 1:length(files)
    
    Xw = Ewhiten*Pall{sp}{i};
    if mod(size(Xw,2)-ws+1, spacing)~=0
            n = mod(size(Xw,2)-ws+1, spacing);
            Xw(:, 1:floor(n/2), :) = [];
            Xw(:, end-ceil(n/2)+1:end, :) = [];
    end
    Xwtr = Xw';
    Xwtr = reshape(Xwtr, [size(Xwtr,1), 1, size(Xwtr,2)]);

    %получаем выход с 1 слоя
        sends = tirbm_inference_fixconv_1d(Xwtr, W, hbias_vec, pars);
        [~,~,~, hpc] = tirbm_sample_multrand_1d(sends, spacing);
        %out1{sp}=hpc;
        tmp{i}=hpc;
        %обрезаем, чтобы данные соответствовали количеству spacing
        if mod(size(hpc,1)-ws+1, spacing)~=0
            n = mod(size(hpc,1)-ws+1, spacing);
            hpc(1:floor(n/2), :,  :) = [];
            hpc(end-ceil(n/2)+1:end, :,  :) = [];
        end
     %получаем выход с 2 слоя
        sends = tirbm_inference_fixconv_1d(hpc, W2, hbias_vec2, pars);
        [~,~,~, hpc2] = tirbm_sample_multrand_1d(sends, spacing);
        tmp2{i}=hpc2;
        %out2{i}=hpc2;
        %обрезаем, чтобы данные соответствовали количеству spacing
        if mod(size(hpc2,1)-ws+1, spacing)~=0
            n = mod(size(hpc2,1)-ws+1, spacing);
            hpc2(1:floor(n/2), :,  :) = [];
            hpc2(end-ceil(n/2)+1:end, :,  :) = [];
        end
        %получаем выход с 3 слоя
        sends = tirbm_inference_fixconv_1d(hpc2, W3, hbias_vec3, pars);
        %[~,~,~, out3{i}] = tirbm_sample_multrand_1d(sends, spacing);
        [~,~,~, tmp3{i}] = tirbm_sample_multrand_1d(sends, spacing);
        
    end; 
    out1{sp}=tmp;
    out2{sp}=tmp2;
    out3{sp}=tmp3;
    
    end;
    
    for sp=1:length(speakers)
    for i=1:length(files)
        out1{sp}{i}=squeeze(out1{sp}{i})';
        out2{sp}{i}=squeeze(out2{sp}{i})';
        out3{sp}{i}=squeeze(out3{sp}{i})';
    end;
    end;
    
    save('E:\temp\123\data\DBN\sp1.mat','out1');
    save('E:\temp\123\data\DBN\sp2.mat','out2');
    save('E:\temp\123\data\DBN\sp3.mat','out3');
	   
   %%
   
%Make features for test
    speakers = [16:25,41:50];
    files = [41:50];
    d = 'E:\temp\123\';
    flist = {};
    
    for sp=1:length(speakers)
        for file=1:length(files)
            flist{length(flist)+1} = strcat(d,int2str(speakers(sp)),' (',int2str(files(file)),').wav');
        end;
    end;

    Pall = {};
    parfor i= 1:length(flist)
        if mod(i,10)==0, fprintf('.'); end
        if mod(i,1000)==0, fprintf('\n'); end
         % read wav file
        [y,fs,~]=readwav(flist{i});
        % convert to spectrogram
        %Pall{i} = get_spectrogram_orig(y, 0, fs);
        nfft = ceil(fs*0.02);
        WINDOW = hamming(nfft);
        noverlap = nfft-ceil(fs*0.01);

        [Pall{i}, ~] = spectrogram(y, WINDOW, noverlap, nfft, fs);
        Pall{i} = flipud(abs(Pall{i}));
        Pall{i} = log(1e-5+Pall{i}) - log(1e-5);
        Pall{i} = Pall{i} - mean(mean(Pall{i}));
        Pall{i} = Pall{i} / sqrt(mean(mean(Pall{i}.^2)));

    end
    
    ws=pars.ws;
    spacing = pars.spacing;
    out1 = cell(1,length(flist));
    out2 = cell(1,length(flist));
    out3 = cell(1,length(flist));
    
    parfor i= 1:length(flist)
    
    Xw = Ewhiten*Pall{i};
    if mod(size(Xw,2)-ws+1, spacing)~=0
            n = mod(size(Xw,2)-ws+1, spacing);
            Xw(:, 1:floor(n/2), :) = [];
            Xw(:, end-ceil(n/2)+1:end, :) = [];
    end
    Xwtr = Xw';
    Xwtr = reshape(Xwtr, [size(Xwtr,1), 1, size(Xwtr,2)]);

    %получаем выход с 1 слоя
        sends = tirbm_inference_fixconv_1d(Xwtr, W, hbias_vec, pars);
        [~,~,~, hpc] = tirbm_sample_multrand_1d(sends, spacing);
        out1{i}=hpc;
        %обрезаем, чтобы данные соответствовали количеству spacing
        if mod(size(hpc,1)-ws+1, spacing)~=0
            n = mod(size(hpc,1)-ws+1, spacing);
            hpc(1:floor(n/2), :,  :) = [];
            hpc(end-ceil(n/2)+1:end, :,  :) = [];
        end
     %получаем выход с 2 слоя
        sends = tirbm_inference_fixconv_1d(hpc, W2, hbias_vec2, pars);
        [~,~,~, hpc2] = tirbm_sample_multrand_1d(sends, spacing);
        out2{i}=hpc2;
        %обрезаем, чтобы данные соответствовали количеству spacing
        if mod(size(hpc2,1)-ws+1, spacing)~=0
            n = mod(size(hpc2,1)-ws+1, spacing);
            hpc2(1:floor(n/2), :,  :) = [];
            hpc2(end-ceil(n/2)+1:end, :,  :) = [];
        end
        %получаем выход с 3 слоя
        sends = tirbm_inference_fixconv_1d(hpc2, W3, hbias_vec3, pars);
        [~,~,~, out3{i}] = tirbm_sample_multrand_1d(sends, spacing);
        
     
    end;
    
     
    for i=1:length(flist)
        out1{i}=squeeze(out1{i})';
        out2{i}=squeeze(out2{i})';
        out3{i}=squeeze(out3{i})';
    end;
    
    trials = zeros(length(speakers)*length(flist),2);
    labels = zeros(length(speakers)*length(flist),1);
    
    for sp=1:length(speakers)
        for file=1:length(flist)
            trials((sp-1)*length(flist)+file,1) = sp;
            trials((sp-1)*length(flist)+file,2) = file;
            if (fix((file-1)/length(files)) == sp-1) 
                labels((sp-1)*length(flist)+file) = 1; 
            end;
        end;
    end;
    
    save('E:\temp\123\data\DBN\tst1.mat','out1');
    save('E:\temp\123\data\DBN\tst2.mat','out2');
    save('E:\temp\123\data\DBN\tst3.mat','out3');
    save('E:\temp\123\data\DBN\labels.mat','trials','labels');