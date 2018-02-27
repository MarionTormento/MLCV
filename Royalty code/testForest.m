function [classPred] = testForest(param, data_test, leaves, nodes, displayClassDistr, displayMap)

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
        
        if displayClassDistr == 1
            figure()
            subplot(2,1,1)
                colormap('jet')
                bar(testProb', 'stacked')
                title(['Predicted Class = ' num2str(classPred(ii))])
                xlabel('Class')
                ylabel('Stacked Tree Probability')
            subplot(2,1,2)
                bar(meanTestProb)
                title(['Predicted Class = ' num2str(classPred(ii))])
                xlabel('Class')
                ylabel('Overall Probability')
        end

    end

    if displayMap == 1    
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
    end

end