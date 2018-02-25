clear all
close all

[data_train, data_test] = getData('Toy_Spiral');

%% Bagging

param.n = 4; %number of bags, n
param.s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
param.replacement = 1; % 0 for no replacement and 1 for replacement

% bagging and visualise bags, Choose a bag for root node.
[bags] = bagging(param, data_train);
visBags(bags, param.replacement);

%% Training Tree

disp('Your Lord and Saviour is training the tree...')
tic

param.numfunct = 3;
param.numlevels = 6;
param.rho = 0.8;

[leaves, nodes] = trainTree(bags, param);
t = toc;
formatSpec = '... and on the %2.2f second, the Lord said "Let there be a Randomised Forest Tree"';
fprintf(formatSpec,t)

%% Test Tree

classPred = [];

for ii = 1:length(data_test)
    
    testProb = zeros(param.n,3);
    
    for k = 1:param.n
        i = 1;
        for j = 1:param.numlevels
            %check if either k,j+1,2*i-1 or k,j+1,2*i is a leaf.
            % IF leaf, get distribution and break from parfor loop.
            if nodes{1,k}{j,i}.x1 == 'Leaf'
                idLeaf = find(leaves(:,1) == k & leaves(:,2) == j &...
                    leaves(:,3) == i);
                testProb(k,:) = leaves(idLeaf(1),4:6);
                break
            else
            % ELSE       
                %IF result from split node = 0, send to k,j+1,2*i-1
                if nodes{1,k}{j,i}.x1 == 'X'
                    if data_test(ii,1) < nodes{1,k}{j,i}.x2
                        i = 2*i - 1;
                    else 
                        i = 2*i;
                    end
                elseif nodes{1,k}{j,i}.x1 == 'Y'
                    if data_test(ii,2) < nodes{1,k}{j,i}.x2
                        i = 2*i - 1;
                    else 
                        i = 2*i;
                    end
                else
                    if (data_test(ii,2) - nodes{1,k}{j,i}.x2 - nodes{1,k}{j,i}.x1*data_test(ii,1)) < 0
                        i = 2*i - 1;
                    else 
                        i = 2*i;
                    end
                end 
            end
        end
    end
    
    meanTestProb = sum(testProb)/size(testProb,1);
    [maxProb, idMax] = max(meanTestProb);
    classPred(ii,1) = idMax;
    
end

figure()
for ii = 1:length(classPred)   
    if classPred(ii) == 1
        plot(data_test(ii,1),data_test(ii,2),'or')
        hold on
    elseif classPred(ii) == 2
        plot(data_test(ii,1),data_test(ii,2),'ob')
        hold on
    else
        plot(data_test(ii,1),data_test(ii,2),'og')
        hold on
    end
end