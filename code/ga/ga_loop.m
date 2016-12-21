%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML122
% Project Title: Feature Selection using GA (Variable Number of Features)
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

%%clc;
%%clear;
%%close all;

%% Main Loop

for it=25:34
    tic;
    disp(['Starting Iteration ' num2str(it) ' ...']);
    
    P=exp(-beta*Costs/WorstCost);
    P=P/sum(P);
    
    % Crossover
    popc=repmat(empty_individual,nc/2,2);
    for k=1:nc/2
        
        % Select Parents Indices
        i1=RouletteWheelSelection(P);
        i2=RouletteWheelSelection(P);

        % Select Parents
        p1=pop(i1);
        p2=pop(i2);
        
        % Apply Crossover
        [popc(k,1).Position, popc(k,2).Position]=Crossover(p1.Position,p2.Position);
        
        % Evaluate Offsprings
        featCol = find(popc(k,1).Position);
        core_gmm_ubm;
        popc(k,1).Cost = eer; popc(k,1).Out = dcf1;
        %[popc(k,1).Cost, popc(k,1).Out]=CostFunction(popc(k,1).Position); %!!
        featCol = find(popc(k,2).Position);
        core_gmm_ubm;
        popc(k,2).Cost = eer; popc(k,2).Out = dcf1;
        %[popc(k,2).Cost, popc(k,2).Out]=CostFunction(popc(k,2).Position); %!!
        
    end
    popc=popc(:);
    
    
    % Mutation
    popm=repmat(empty_individual,nm,1);
    for k=1:nm
        
        % Select Parent
        i=randi([1 nPop]);
        p=pop(i);
        
        % Apply Mutation
        popm(k).Position=Mutate(p.Position,mu);
        
        % Evaluate Mutant
        featCol = find(popm(k).Position);
        core_gmm_ubm;
        popm(k).Cost = eer; popm(k).Out = dcf1;
        %[popm(k).Cost, popm(k).Out]=CostFunction(popm(k).Position); %!!
        
    end
    
    % Add best top nmb feature mutants
    popmb=repmat(empty_individual,nbm,1);
    for k=1:nbm
        
        p=pop(k);
        
        % Apply Mutation
        popmb(k).Position=Mutate(p.Position,mu);
        
        % Evaluate Mutant
        featCol = find(popmb(k).Position);
        core_gmm_ubm;
        popmb(k).Cost = eer; popmb(k).Out = dcf1;
        %[popm(k).Cost, popm(k).Out]=CostFunction(popm(k).Position); %!!
        
    end
    
    % Create Merged Population
    pop=[pop
         popc
         popm
         popmb]; %#ok
     
    % Find similar genes
    
    toDelete = [];
    for p1 =1:length(pop)-1
        
        for p2 = p1+1:length(pop)
            if isequal(pop(p1).Position,pop(p2).Position)
                toDelete = [toDelete p1];
                break;
            end
        end
    end
    pop(toDelete) = []; 
    % Sort Population 
    % AGain !!!
    Costs=[pop.Cost;pop.Out]';
    Costs1=round(Costs*100000); 
    [Costs1, SortOrder]=sortrows(Costs1);
    pop=pop(SortOrder);
    Costs = Costs1/100000;
    
    % Update Worst Cost
    WorstCost=max(WorstCost,pop(end).Cost);
    
    % Truncation
    pop=pop(1:nPop);
    Costs=Costs(1:nPop);
    
    % Store Best Solution Ever Found
    BestSol=pop(1);
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    toc
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
   
end

%% Results

figure;
plot(BestCost,'LineWidth',2);
ylabel('Cost');
