function [classPred] = testForest(param, data_test, leaves, nodes, displayClassDistr, displayMap)

classPred = [];

for idx = 1:size(data_test,1)
    
    testProb = zeros(param.n,10);
    
    for k = 1:param.n
        i = 1;
        for j = 1:param.numlevels
            if nodes{1,k}{j,i}.x1 == 'Leaf'
                idLeaf = find(leaves(:,1) == k & leaves(:,2) == j &...
                    leaves(:,3) == i);
                testProb(k,:) = leaves(idLeaf(1),4:end
                % histcountid(idleaf) = histcoutid(idleaf) + 1;
                break
            else
                if nodes{1,k}{j,i}.x1 == 'X'
                    if data_test(idx,nodes{1,k}{j,i}.dim(1)) < nodes{1,k}{j,i}.x2
                        i = 2*i - 1;
                    else
                        i = 2*i;
                    end
                elseif nodes{1,k}{j,i}.x1 == 'Y'
                    if data_test(idx,nodes{1,k}{j,i}.dim(2)) < nodes{1,k}{j,i}.x2
                        i = 2*i - 1;
                    else
                        i = 2*i;
                    end
                else
                    if (data_test(idx,nodes{1,k}{j,i}.dim(2)) - (nodes{1,k}{j,i}.x1*data_test(idx,nodes{1,k}{j,i}.dim(1)) + nodes{1,k}{j,i}.x2)) < 0
                        i = 2*i - 1;
                    else
                        i = 2*i;
                    end
                end
            end
        end
    end
    
    meanTestProb = sum(testProb,1)/size(testProb,1);
    [maxProb, idMax] = max(meanTestProb);
    classPred(idx,1) = idMax;
    classPred(idx,2:11) = meanTestProb;
    
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