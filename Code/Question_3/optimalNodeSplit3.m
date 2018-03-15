% compute the optimal node split between two type of weak learner
% axis-aligned and linear
function [childrenBest, infoGainBest] = optimalNodeSplit3(param, rootNode) 
numfunct = param.numfunct;
diff = 0;

[axisCh, axisInfo] = axisNodeSplit3(param, rootNode, numfunct);
[linearCh, linearInfo] = linearNodeSplit3(param, rootNode, numfunct);

[maxInfo idxInfo] = max([axisInfo.Gain, linearInfo.Gain]); 
if idxInfo == 1
    childrenBest = axisCh;
    infoGainBest = axisInfo;
elseif idxInfo == 2
    childrenBest = linearCh;
    infoGainBest = linearInfo;
end

end
