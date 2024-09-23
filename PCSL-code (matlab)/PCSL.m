function [DAG, time] = PCSL(Data_0, Alpha, rand_sample_numb)
% by XianjieGuo 2022.8.31

% input:
%     Data_0: the values of each variable in the data matrix start from 1.
%     Alpha: significance level, e.g., 0.01 or 0.05.
%     rand_sample_numb: the number of sub-datasets generated.
% output:
%     DAG: a directed acyclic graph learned on a given dataset.
%     time: the runtime of the algorithm.

ns_D0=max(Data_0);
[~,p]=size(Data_0);
maxK=3;

start=tic;

PCs=cell(1,p);
skeleton=zeros(p,p);
for i=1:p
    [pc,~,~,~]=HITONPC_G2_CI0(Data_0,i,Alpha,ns_D0,p,maxK);
    PCs{i}=pc;
    skeleton(i,pc)=1;
end

conflicts_node_pairs={};
conflicts_nodes=[];
for i=1:p
    for j=1:i-1
        if skeleton(i,j)~=skeleton(j,i)
            conflicts_node_pairs{end+1}=[i,j];
            conflicts_nodes=[conflicts_nodes i j];
        end
    end
end

tolerance=3; 
Data_Score_Mean=-1; 
Max_Data_Bootstrap_N=cell(1,rand_sample_numb); 

while(tolerance>0&&~isempty(conflicts_nodes))
    results_D_n=cell(2,length(conflicts_node_pairs));
    weights_D_n=cell(2,length(conflicts_node_pairs));
    
    [Data_Bootstrap_N]=gen_bootstrap_data(Data_0,rand_sample_numb);
    Data_Score_N=zeros(1,rand_sample_numb); 
    
    PCs_conflicting=PCs(conflicts_nodes);
    
    for i=1:rand_sample_numb
        learned_PCs_conflicting=cell(1,length(conflicts_nodes));
        
        ns=max(Data_Bootstrap_N{i});
        for j=1:length(conflicts_node_pairs)
            [pc1,~,~,~]=HITONPC_G2_CI1(Data_Bootstrap_N{i},conflicts_node_pairs{j}(1),Alpha,ns,p,maxK);
            [F1,~,~,~]=eva_PC(pc1,PCs{conflicts_node_pairs{j}(1)});
            weights_D_n{1,j}=[weights_D_n{1,j} F1];
            if ismember(conflicts_node_pairs{j}(2),pc1)
                results_D_n{1,j}=[results_D_n{1,j} 1];
            else
                results_D_n{1,j}=[results_D_n{1,j} -1];
            end
            learned_PCs_conflicting{1+(j-1)*2}=pc1;
            
            [pc2,~,~,~]=HITONPC_G2_CI1(Data_Bootstrap_N{i},conflicts_node_pairs{j}(2),Alpha,ns,p,maxK);
            [F1,~,~,~]=eva_PC(pc2,PCs{conflicts_node_pairs{j}(2)});
            weights_D_n{2,j}=[weights_D_n{2,j} F1];
            if ismember(conflicts_node_pairs{j}(1),pc2)
                results_D_n{2,j}=[results_D_n{2,j} 1];
            else
                results_D_n{2,j}=[results_D_n{2,j} -1];
            end
            learned_PCs_conflicting{2+(j-1)*2}=pc2;
        end
        
        [F1_Di,~,~,~]=eva_Dataset(learned_PCs_conflicting,PCs_conflicting);
        Data_Score_N(i)=F1_Di;
    end
    
    aux=mean(Data_Score_N);
    if aux>Data_Score_Mean
        Data_Score_Mean=aux;
        
        Score=ones(1,length(conflicts_node_pairs))*(-100);
        for i=1:length(conflicts_node_pairs)
            Score(i)=(sum(results_D_n{1,i}.*weights_D_n{1,i})+sum(results_D_n{2,i}.*weights_D_n{2,i}))/(2*rand_sample_numb);
        end
        
        [PCs] = update_PCs(conflicts_node_pairs,PCs,Score);
        
        Max_Data_Bootstrap_N=Data_Bootstrap_N;
    else
        tolerance=tolerance-1;
    end
end

skeleton=zeros(p,p);
for i=1:p
    skeleton(i,PCs{i})=1;
end

cpm = tril(sparse(skeleton));

if isempty(Max_Data_Bootstrap_N{1})
    [Max_Data_Bootstrap_N]=gen_bootstrap_data(Data_0,rand_sample_numb);
end
Sub_DAGs=cell(1,rand_sample_numb);
for i=1:rand_sample_numb
    LocalScorer = bdeulocalscorer(Max_Data_Bootstrap_N{i}, max(Max_Data_Bootstrap_N{i}));
    HillClimber = hillclimber(LocalScorer, 'CandidateParentMatrix', cpm);
    Sub_DAGs{i} = HillClimber.learnstructure();
end

Sub_DAGs_Avg=zeros(p,p);
for i=1:rand_sample_numb
    Sub_DAGs_Avg=Sub_DAGs_Avg+Sub_DAGs{i};
end
Sub_DAGs_Avg=Sub_DAGs_Avg/rand_sample_numb;

Sub_DAGs_Avg(Sub_DAGs_Avg>=0.5)=1;
Sub_DAGs_Avg(Sub_DAGs_Avg<0.5)=0;

DAG=Sub_DAGs_Avg;

time=toc(start);

end