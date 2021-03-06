%%set needed features in config file
load_cfg;
maxSteps = 50; %% �������� ����� ������� ���������
method_ch_steps = 3; %% ���������� �����, � ������ �� ���������� ������ ����� ������� �������� �����
current_steps = 0; %% ������� ���������� �����

featMax = 97; %% ������������ ���������� ��������� ?????

bestFeat = 1:maxSteps; %% ������ ���������, ��������� �� i-�� ���� (����������� ��� ���������)
indexes = zeros(1,maxSteps); %% indexes of the best features



%%first greedy method
%%adding one feature at a time

oneFeatError = 100*ones(maxSteps,featMax);
oneFeatDcf = 10000*ones(maxSteps,featMax);

method = 0; % 0 - add, 1 - del.
featSize = 0; %% current size of feature vector

%%for tessting paper featCol 14MFC + Vp.

featCol = [1     2     3     4     5     6     7     8     9    10    11    13    14    26    29    30    32    33    71    89];
featSize = size(featCol,2);
indexes = [featCol zeros(1,maxSteps)]; %% indexes of the best features

for i = featSize+1 : maxSteps,
    tic;
    if method == 0 
        fea_add;
    else
        fea_del;
    end
    toc
    
       %%output result
       if (i > 1)
            fprintf('Result of previous step, EER= %6.3f, dcf= %6.3f \n',oneFeatError(i-1,bestFeat(i-1)),oneFeatDcf(i-1,bestFeat(i-1)));
       end
       fprintf('Result of  step � %i, EER= %6.3f, dcf= %6.3f \n',i,M,oneFeatDcf(i,I));

    if (i == 1) || (oneFeatError(i,I)>oneFeatError(i-1,bestFeat(i-1))) % ���� � ��� ������ �� �����������, �� ����������� �������
       current_steps = current_steps + 1;
    else
        if (oneFeatError(i,I)<oneFeatError(i-1,bestFeat(i-1)))
        current_steps = 0;
        end
    end
    
    if (current_steps == method_ch_steps+1 && method == 1) % ���� ������� ���, ��� ������, �����������
        break;
    end
    
    if (current_steps == method_ch_steps && method ~= 1)
        current_steps = 0;
        method = 1;
    end
      
    
end

%%get and save results


