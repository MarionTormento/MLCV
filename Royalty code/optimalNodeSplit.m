
function [childrenBest, infoGainBest] = optimalNodeSplit(param, rootNode) % compute the optimal split node between axis and linear
    
    rho = round(length(rootNode)*param.rho);
    numfunct = param.numfunct;
        
    X = [min(rootNode(:,1)), max(rootNode(:,1))];
    YInt = [min(rootNode(:,2)), max(rootNode(:,2))];
    
    % implementing rho
%     id = randperm(length(rootNode),rho);
%     data = rootNode(id,:);
%     [axisCh, axisInfo] = axisNodeSplit(X(1), X(2), YInt(1), YInt(2), data, numfunct);
%     [linearCh, linearInfo] = linearNodeSplit(YInt(1), YInt(2), data, numfunct);
    
    % not implementing rho
    [axisCh, axisInfo] = axisNodeSplit(X(1), X(2), YInt(1), YInt(2), rootNode, numfunct);
    [linearCh, linearInfo] = linearNodeSplit(YInt(1), YInt(2), rootNode, numfunct);
    
    [maxInfo idxInfo] = max([axisInfo.Gain, linearInfo.Gain]); %if idxInfo return 1 => Axis, 2 => linear
    if idxInfo == 1 
            childrenBest = axisCh;
            infoGainBest = axisInfo;
    elseif idxInfo == 2
            childrenBest = linearCh;
            infoGainBest = linearInfo;
    end
end
    