
function [childrenBest, infoGainBest] = optimalNodeSplit3(param, rootNode) % compute the optimal split node between axis and linear

numfunct = param.numfunct;
diff = 0;

while diff == 0
    randomDim = randi(param.dimensions,[1,2]);
    
    X = [min(rootNode(:,randomDim(1,1))), max(rootNode(:,randomDim(1,1)))];
    YInt = [min(rootNode(:,randomDim(1,2))), max(rootNode(:,randomDim(1,2)))];
    if X(1) ~= X(2) && YInt(1) ~= YInt(2)
        diff = 1;
    end
end
[axisCh, axisInfo] = axisNodeSplit3(X(1), X(2), YInt(1), YInt(2), param, rootNode, numfunct,randomDim);
[linearCh, linearInfo] = linearNodeSplit3(YInt(1), YInt(2), param, rootNode, numfunct, randomDim);

[maxInfo idxInfo] = max([axisInfo.Gain, linearInfo.Gain]); %if idxInfo return 1 => Axis, 2 => linear
if idxInfo == 1
    childrenBest = axisCh;
    infoGainBest = axisInfo;
elseif idxInfo == 2
    childrenBest = linearCh;
    infoGainBest = linearInfo;
end
infoGainBest.dim = randomDim;

end
