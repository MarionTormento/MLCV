function [histCountID] = testCodebook(param, data_test, leaves, nodes)

histCountID = zeros(1,size(leaves,1));

for idx = 1:size(data_test,1)
    
    for k = 1:param.n
        i = 1;
        for j = 1:param.numlevels
            if nodes{1,k}{j,i}.x1 == 'Leaf'
                idLeaf = find(leaves(:,1) == k & leaves(:,2) == j &...
                    leaves(:,3) == i);
                histCountID(1, idLeaf) = histCountID(1, idLeaf) + 1;
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
    
end