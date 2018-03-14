function binCount = leafTest(param, rootNode)
    for i = 1:10
        bin(i,1) = isempty(rootNode((rootNode(:,param.dimensions+1) == i) == 1,:));
    end
    if sum(bin,1) > 8 || (size(rootNode,1) < 10)
        binCount = 1;
    else
        binCount = 0;
    end
end
