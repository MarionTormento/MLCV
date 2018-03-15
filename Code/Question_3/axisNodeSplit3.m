% compute the accuracy of the training data
function [childrenBest, infoGainBest] = axisNodeSplit3(param, rootNode, numfunct) % Compute the best 'x=...' split node for the bag

infoGainBest.x1 = 'X';
infoGainBest.x2 = 0;
infoGainBest.Gain = 0;
childrenBest = [];
diff = 0;

% Axis Split Function for x=i
linSplitThreshold.x1 = 'X';
for i = 1:numfunct
    while diff == 0
        randomDim = randi(param.dimensions,[1,2]);
        minX = min(rootNode(:,randomDim(1,1)));
        maxX = max(rootNode(:,randomDim(1,1)));
        minY = min(rootNode(:,randomDim(1,2)));
        maxY = max(rootNode(:,randomDim(1,2)));
        if minX ~= maxX
            diff = 1;
        end
    end
    randomSampX = randperm(round((maxX-minX)/0.01),numfunct);
    threshold = minX + 0.01*randomSampX(i);
    linSplitThreshold.x2 = threshold;
    [children, infoGain] = childrenAndInfo3(param, rootNode, linSplitThreshold,randomDim);
    
    if infoGain > infoGainBest.Gain
        infoGainBest.x2 = threshold;
        infoGainBest.Gain = infoGain;
        infoGainBest.dim = randomDim;
        childrenBest = children;
    end
    diff = 0;
end

% Axis Split Function for y=i
linSplitThreshold.x1 = 'Y';
diff = 0;

for i = 1:numfunct
    while diff == 0
        randomDim = randi(param.dimensions,[1,2]);
        minX = min(rootNode(:,randomDim(1,1)));
        maxX = max(rootNode(:,randomDim(1,1)));
        minY = min(rootNode(:,randomDim(1,2)));
        maxY = max(rootNode(:,randomDim(1,2)));
        if minY ~= maxY
            diff = 1;
        end
    end
    randomSampY = randperm(round((maxY-minY)/0.01),numfunct);
    threshold = minY + 0.01*randomSampY(i);
    linSplitThreshold.x2 = threshold;
    [children, infoGain] = childrenAndInfo3(param, rootNode, linSplitThreshold,randomDim);
    if infoGain > infoGainBest.Gain
        infoGainBest.x1 = 'Y';
        infoGainBest.x2 = threshold;
        infoGainBest.Gain = infoGain;
        infoGainBest.dim = randomDim;
        childrenBest = children;
    end
    diff = 0;
end

clear children
clear infoGain
clear linSplitThreshold
end
