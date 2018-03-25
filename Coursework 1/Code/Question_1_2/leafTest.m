% test to know if the node is a leaf
function binCount = leafTest(rootNode)
    for i = 1:3
        bin(i,1) = isempty(rootNode((rootNode(:,3) == i) == 1,:));
    end
    if sum(bin,1) > 1 || (length(rootNode) < 10)
        binCount = 1;
    else
        binCount = 0;
    end
end
