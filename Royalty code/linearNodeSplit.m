
function [childrenBest, infoGainBest] = linearNodeSplit(minYInt, maxYInt, rootNode, numfunct) % Compute the best "y=mx+p" split node for the bag
    
    infoGainBest.x1 = 0;
    infoGainBest.x2 = 0;
    infoGainBest.Gain = 0;
    childrenBest = [];
    
    randomSampYInt = randperm(round(abs(maxYInt - minYInt)/0.001),numfunct);
    %given an y intercept, calculate good max and min gradients
    if rootNode(rootNode(:,2) == min(rootNode(:,2)),1) < 0
        maxGrad = (randomSampYInt - min(rootNode(:,2))/0.001)*0.001./(-rootNode(rootNode(:,2) == min(rootNode(:,2)),1));
        minGrad = (randomSampYInt - max(rootNode(:,2))/0.001)*0.001./(-rootNode(rootNode(:,2) == max(rootNode(:,2)),1));
    else 
        minGrad = (randomSampYInt - min(rootNode(:,2))/0.001)*0.001./(-rootNode(rootNode(:,2) == min(rootNode(:,2)),1));
        maxGrad = (randomSampYInt - max(rootNode(:,2))/0.001)*0.001./(-rootNode(rootNode(:,2) == max(rootNode(:,2)),1));
    end
    %Linear Split Function y = m*x+p
    for m = 1:numfunct
        %for each yint, calc 3 random grads between that points man and min
        %grad
        randomSampGrad(m,:) = minGrad/0.001 + randperm(round((abs(maxGrad(m)+minGrad(m))/0.001)),numfunct);
        for p = 1:numfunct
            linSplitThreshold.x1 = randomSampGrad(m,p)*0.001;
            linSplitThreshold.x2 = randomSampYInt(p)*0.001;
            [children, infoGain] = childrenAndInfo(rootNode, linSplitThreshold);
            if infoGain > infoGainBest.Gain
                 infoGainBest.x1 = randomSampGrad(m,p)*0.001;
                 infoGainBest.x2 = randomSampYInt(p)*0.001;
                 infoGainBest.Gain = infoGain;
                 childrenBest = children;
            end
         end
    end
end
