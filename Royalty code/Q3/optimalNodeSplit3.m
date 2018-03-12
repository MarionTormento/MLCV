
function [childrenBest, infoGainBest] = optimalNodeSplit3(param, rootNode) % compute the optimal split node between axis and linear

numfunct = param.numfunct;
diff = 0;

[axisCh, axisInfo] = axisNodeSplit3(param, rootNode, numfunct);
[linearCh, linearInfo] = linearNodeSplit3(param, rootNode, numfunct);

[maxInfo idxInfo] = max([axisInfo.Gain, linearInfo.Gain]); %if idxInfo return 1 => Axis, 2 => linear
if idxInfo == 1
    childrenBest = axisCh;
    infoGainBest = axisInfo;
elseif idxInfo == 2
    childrenBest = linearCh;
    infoGainBest = linearInfo;
end

end
