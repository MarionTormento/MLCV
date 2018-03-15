% test the RF and compute the predicted class
function [classPred] = testForest(param, data_test, leaves, nodes, displayClassDistr, displayMap)

%initialise output
classPred = [];

%for each data point
for idx = 1:length(data_test)
    
    %initialise the test probability
    testProb = zeros(param.n,3);
    
    % send the data point down the tree
    for k = 1:param.n
        i = 1;
        for j = 1:param.numlevels
            % if the data point has arrived at a leaf, get this leafs
            % class probability distribution
            if nodes{1,k}{j,i}.x1 == 'Leaf'
                idLeaf = find(leaves(:,1) == k & leaves(:,2) == j &...
                    leaves(:,3) == i);
                testProb(k,:) = leaves(idLeaf(1),4:6);
                break
            else
                %Else determine where the datapoint goes (left or right) as
                %a result of this nodes split function
                if nodes{1,k}{j,i}.x1 == 'X'
                    if data_test(idx,1) < nodes{1,k}{j,i}.x2
                        i = 2*i - 1;
                    else
                        i = 2*i;
                    end
                elseif nodes{1,k}{j,i}.x1 == 'Y'
                    if data_test(idx,2) < nodes{1,k}{j,i}.x2
                        i = 2*i - 1;
                    else
                        i = 2*i;
                    end
                else
                    if (data_test(idx,2) - (nodes{1,k}{j,i}.x1*data_test(idx,1) + nodes{1,k}{j,i}.x2)) < 0
                        i = 2*i - 1;
                    else
                        i = 2*i;
                    end
                end
            end
        end
    end
    
    %Calculate the normalised class probability distribution across all
    %trees for this data point and find its most likely class label.
    meanTestProb = sum(testProb,1)/size(testProb,1);
    [maxProb, idMax] = max(meanTestProb);
    classPred(idx,1) = idMax;
    classPred(idx,2:4) = meanTestProb;
    
    %Plotting
    if displayClassDistr == 1
        figure()
        subplot(2,1,1)
        ColorSet = varycolor(length(testProb));
        H = bar(testProb', 'stacked');
        for k=1:length(testProb)
            set(H(k),'facecolor',ColorSet(k, :))
        end
        title(['Predicted Class = ' num2str(classPred(idx))])
        xlabel('Class')
        ylabel('Stacked Tree Probability')
        subplot(2,1,2)
        bar(meanTestProb)
        title(['Predicted Class = ' num2str(classPred(idx))])
        xlabel('Class')
        ylabel('Overall Probability')
    end
    
end

if displayMap == 1
    figure
        plot(data_test(classPred(:,1) == 1,1),data_test(classPred(:,1) == 1,2),'or')
        hold on
        plot(data_test(classPred(:,1) == 2,1),data_test(classPred(:,1) == 2,2),'ob')
        hold on
        plot(data_test(classPred(:,1) == 3,1),data_test(classPred(:,1) == 3,2),'og')
        title({[num2str(param.n) ' Trees'],[num2str(param.numlevels) ' levels'],...
            [num2str(param.numfunct) ' split functions'], ...
            ['Training Time ' num2str(param.trainingtime) ' s']})
    
    figure
        for ii = 1:length(data_test)
            plot(data_test(ii,1), data_test(ii,2), 'o', 'Color', [classPred(ii,2) classPred(ii,4) classPred(ii,3)])
            hold on
        end
        title({[num2str(param.n) ' Trees'],[num2str(param.numlevels) ' levels'],...
            [num2str(param.numfunct) ' split functions'], ...
            ['Training Time ' num2str(param.trainingtime) ' s']})
end

end