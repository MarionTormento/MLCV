% Split node for an axis split weak learner 'x=...' or 'y=...'
function [childrenBest, infoGainBest] = axisNodeSplit(minX, maxX, minY, maxY, rootNode, numfunct) % Compute the best 'x=...' split node for the bag

infoGainBest.x1 = 'X';
infoGainBest.x2 = 0;
infoGainBest.Gain = 0;
childrenBest = [];

randomSampX = randperm(round((maxX-minX)/0.0001),numfunct);

% Axis Split Function for x=i
linSplitThreshold.x1 = 'X';
for i = 1:numfunct
    threshold = minX + 0.0001*randomSampX(i);
    linSplitThreshold.x2 = threshold;
    [children, infoGain] = childrenAndInfo(rootNode, linSplitThreshold);
    if infoGain > infoGainBest.Gain
        infoGainBest.x2 = threshold;
        infoGainBest.Gain = infoGain;
        childrenBest = children;
    end
end

% Axis Split Function for y=i
linSplitThreshold.x1 = 'Y';
randomSampY = randperm(round((maxY-minY)/0.0001),numfunct);
for i = 1:numfunct
    threshold = minY + 0.0001*randomSampY(i);
    linSplitThreshold.x2 = threshold;
    [children, infoGain] = childrenAndInfo(rootNode, linSplitThreshold);
    if infoGain > infoGainBest.Gain
        infoGainBest.x1 = 'Y';
        infoGainBest.x2 = threshold;
        infoGainBest.Gain = infoGain;
        childrenBest = children;
    end
end

clear children
clear infoGain
clear linSplitThreshold
end
