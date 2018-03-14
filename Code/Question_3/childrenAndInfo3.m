function [outputnodes, infoGain] = childrenAndInfo3(param, inputnode, linSplitThreshold,randomDim)

m = linSplitThreshold.x1;
p = linSplitThreshold.x2;

if m == 'X' % Axis aligned x = p
    index = (inputnode(:,randomDim(1,1)) - p) > 0 ;
elseif m == 'Y' %axis aligned y = p
    index = (inputnode(:,randomDim(1,2)) - p) > 0 ; 
else % Axis aligned y = p and linear function y = m*x+p
    index = (inputnode(:,randomDim(1,2)) - p - m*inputnode(:,randomDim(1,1))) > 0 ;
end
outputnodes{1} = inputnode(index == 0,:);
outputnodes{2} = inputnode(index == 1,:);

clear m
clear p
clear index

infoGain = computeInfo3(param, inputnode, outputnodes);

end % return the ouput nodes for desired threhold, and the corresponding info gain
