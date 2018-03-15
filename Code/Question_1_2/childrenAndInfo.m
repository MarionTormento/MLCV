% Compute the two children from the inputNode knowing the split function
% return the ouput nodes for desired threhold, and the corresponding info
% gain
function [outputnodes, infoGain] = childrenAndInfo(inputnode, linSplitThreshold)

m = linSplitThreshold.x1;
p = linSplitThreshold.x2;

if m == 'X' % Axis aligned x = p
    index = (inputnode(:,1) - p) > 0 ;
elseif m == 'Y' %axis aligned y = p
    index = (inputnode(:,2) - p) > 0 ; 
else % Linear function y = m*x+p
    index = (inputnode(:,2) - p - m*inputnode(:,1)) > 0 ;
end
outputnodes{1} = inputnode(index == 0,:);
outputnodes{2} = inputnode(index == 1,:);

clear m
clear p
clear index

infoGain = computeInfo(inputnode, outputnodes);

end 
