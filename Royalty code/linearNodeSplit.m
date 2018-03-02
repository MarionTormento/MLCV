
function [childrenBest, infoGainBest] = linearNodeSplit(minYInt, maxYInt, rootNode, numfunct) % Compute the best "y=mx+p" split node for the bag
    
    infoGainBest.x1 = 0;
    infoGainBest.x2 = 0;
    infoGainBest.Gain = 0;
    childrenBest = [];
    
    randomSampYInt = randperm(round(abs(maxYInt - minYInt)/0.0001),numfunct);
    %given a y intercept, calculate the gradients of the lines to the min
    %and max data points. These gradients are calculated for the 3
    %(numfunct) y intercepts which we try
    Grad1 = (randomSampYInt - min(rootNode(:,2))/0.0001)*0.0001./(-rootNode(rootNode(:,2) == min(rootNode(:,2)),1));
    Grad2 = (randomSampYInt - max(rootNode(:,2))/0.0001)*0.0001./(-rootNode(rootNode(:,2) == max(rootNode(:,2)),1));
    %Linear Split Function y = m*x+p
    for m = 1:numfunct
        %for each yint, calculagte which gradient is larger
    if Grad1(m) > Grad2(m)
        maxGrad = Grad1(m);
        minGrad = Grad2(m);
    else 
        maxGrad = Grad2(m);
        minGrad = Grad1(m);
    end
    % calculate a random gradient in between the min and max possible
        randomSampGrad(m,:) = minGrad/0.0001 + randperm(round((maxGrad-minGrad)/0.0001),numfunct);
        %perform linear split based on this y int and its gradient randomly
        %generated
        for p = 1:numfunct
            linSplitThreshold.x1 = randomSampGrad(m,p)*0.0001;
            linSplitThreshold.x2 = randomSampYInt(p)*0.0001;
            [children, infoGain] = childrenAndInfo(rootNode, linSplitThreshold);
            if infoGain > infoGainBest.Gain
                 infoGainBest.x1 = randomSampGrad(m,p)*0.0001;
                 infoGainBest.x2 = randomSampYInt(p)*0.0001;
                 infoGainBest.Gain = infoGain;
                 childrenBest = children;
            end
         end
    end
end
