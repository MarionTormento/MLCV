clear all
close all

[data_train, data_test] = getData('Toy_Spiral');

%% Bagging

n = 4; %number of bags, n
s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
replacement = 1; % 0 for no replacement and 1 for replacement

infoGain = []; %initialise infoGain

% bagging and visualise bags, Choose a bag for root node.
[bags] = bagging(n, s, data_train, replacement);
%visNodes(bags, replacement, infoGain);

%% Split Function
param.rho = 3;

%% Recursive test
% Branch 0 -> 1 & 2
rootNode = bags{1};
[children, infoGain] = optimalNodeSplit(param, rootNode);
clear rootNode
visNodes2(children, replacement, infoGain);
numlevels = 3;

for j = 1:numlevels   
    for i = 1:length(children)
        rootNode = children{i};
        [childrenNew, infoGain] = optimalNodeSplit(param, rootNode);
        clear rootNode
        visNodes2(childrenNew, replacement, infoGain);
        if exist('childrenNewF', 'var')
            childrenNewF = [childrenNewF, childrenNew];
        else
            childrenNewF = childrenNew;
        end
        pause
    end

    children = childrenNewF;

end

% % Branch 2 -> 21 & 22
% rootNode = children{2};
% [children, infoGain] = optimalNodeSplit(param, rootNode);
% clear rootNode
% visNodes(children, replacement, infoGain);
% 
% % Branch 21 -> 211 & 212
% rootNode = children{1};
% [children, infoGain] = optimalNodeSplit(param, rootNode);
% clear rootNode
% visNodes(children, replacement, infoGain);

%% Plotting
% figure
% for i = 1:length(data_train)
%     if data_train(i,3) == 1
%         plot(data_train(i,1),data_train(i,2),'or')
%         hold on
%     elseif data_train(i,3) == 2
%         plot(data_train(i,1),data_train(i,2),'+b')
%         hold on
%     elseif data_train(i,3) == 3
%         plot(data_train(i,1),data_train(i,2),'*g')
%         hold on
%     end
% end
% grid on

function [bags] = bagging(n, s, data_train, replacement)
    
    if replacement == 1
        %with replacement
        for i = 1:n
            randomIndex = randperm(length(data_train),round(s));
            bags{i} = data_train(randomIndex,:);
        end
    elseif replacement == 0
        %withoutreplacement
        t = round((length(data_train)-1)/n);
        randomIndexTemp = 1:length(data_train);
        for i = 1:n
            randomIndex = randperm(length(randomIndexTemp),t);
            bags{i} = data_train(randomIndex,:);
            randomIndexTemp(randomIndex) = [];
        end
    end
end

function visNodes2(inputs, replacement, infoGain)

if ~iscell(inputs)
    inputscell{1}(:,:) = inputs;
    inputscell{2} =[];
    clear children
    inputs = inputscell;
end

% Plot the position of the toy present in each bag
figure()

if infoGain(1,1) == 'X'
   threshold_y = -1:0.1:1;
   threshold_x = infoGain(1,2)*ones(1,length(threshold_y));
else
   threshold_x = -1:0.1:1; 
   threshold_y = infoGain(1,1).*threshold_x+infoGain(1,2);
end

for i = 1:length(inputs)
    subplot(2,2,1)
    for j = 1:length(inputs{i})
        if inputs{i}(j,3) == 1
            plot(inputs{i}(j,1),inputs{i}(j,2),'or')
            hold on
        elseif inputs{i}(j,3) == 2
            plot(inputs{i}(j,1),inputs{i}(j,2),'+b')
            hold on
        elseif inputs{i}(j,3) == 3
            plot(inputs{i}(j,1),inputs{i}(j,2),'*g')
            hold on
        end
    end
    if ~isempty(infoGain)
        if replacement == 0
            title({['Parent and threshold without replacement,'];['info gain = ' num2str(infoGain(1,3))]})
        elseif replacement == 1
            title({['Parent and threshold with replacement,'];['info gain = ' num2str(infoGain(1,3))]})
        end
    else
        if replacement == 0
            title(['Parent and threshold without replacement'])
        elseif replacement == 1
            title(['Parent and threshold with replacement'])
        end
    end
    xlabel('x co-ordinate')
    ylabel('y co-ordinate')
    plot(threshold_x,threshold_y)
    axis([-1 1 -1 1])
    grid on
end

% Plot the histogram of the toy class repartition in each bag
%figure
for i = 1:length(inputs)
    subplot(2,2,i+2)
    if ~isempty(inputs{i})
        histogram(inputs{i}(:,3))
    end
    xlabel('Category')
    ylabel('# of Occurences')
    if ~isempty(infoGain)
            title({['Child ' num2str(i) ','];['info gain = ' num2str(infoGain(1,3))]})
    else
            title(['Child ' num2str(i) '.'])
    end
    grid on
end

end


function [childrenBest, infoGainBest] = axisNodeSplit(minX, maxX, rootNode, rho) % Compute the best 'x=...' split node for the bag
    infoGainBest = [0,0,0];
    childrenBest = [];
    randomSamp = randperm(round((maxX-minX)/0.1),rho);
    % Axis Split Function for x=i
    for i = 1:rho
        linSplitThreshold = ['X', randomSamp(i)*0.1];
        [children, infoGain] = childrenAndInfo(rootNode, linSplitThreshold);
        if infoGain > infoGainBest(3)
            infoGainBest = [NaN, randomSamp(i)*0.1, infoGain];
            childrenBest = children;
        end
    end
    if isempty(childrenBest) 
        childrenBest = rootNode;
    end
    clear children
    clear infoGain
    clear linSplitThreshold
end

function [childrenBest, infoGainBest] = linearNodeSplit(minGrad, maxGrad, minXInt, maxXInt, rootNode, rho) % Compute the best "y=mx+p" split node for the bag
    
    %n = 1;
    infoGainBest = [0,0,0];
    childrenBest = [];
    randomSampGrad = randperm(round((maxGrad-minGrad)/0.1),rho);
    randomSampInt = randperm(round((maxXInt-minXInt)/0.1),rho);
    %Linear Split Function y = m*x+p
    for m = 1:rho
        %m = 1;
        for p = 1:rho
            linSplitThreshold = [randomSampGrad(m)*0.1, randomSampInt(p)*0.1];
            [children, infoGain] = childrenAndInfo(rootNode, linSplitThreshold);
            if infoGain > infoGainBest(3)
                 infoGainBest = [randomSampGrad(m)*0.1, randomSampInt(p)*0.1,infoGain];
                 childrenBest = children;
            end
            % visNodes(children, replacement, infoGain);
            % m = m + 1;
        end
        %n = n + 1;
    end
    if isempty(childrenBest) 
        childrenBest = rootNode;
    end
end

function [childrenBest, infoGainBest] = optimalNodeSplit(param, rootNode) % compute the optimal split node between axis and linear
    
    rho = param.rho;
    X = [min(rootNode(:,1)), max(rootNode(:,1))];
    Grad = [-3, 3];
    XInt = [min(rootNode(:,1)), max(rootNode(:,1))];
    
    [axisCh, axisInfo] = axisNodeSplit(X(1), X(2), rootNode, rho);
    [linearCh, linearInfo] = linearNodeSplit(Grad(1), Grad(2), XInt(1), XInt(2), rootNode, rho);
    
    [maxInfo idxInfo] = max([axisInfo(1,3), linearInfo(1,3)]) %if idxInfo return 1 => Axis, 2 => linear
    if idxInfo == 1
            childrenBest = axisCh;
            infoGainBest = axisInfo;
    elseif idxInfo == 2
            childrenBest = linearCh;
            infoGainBest = linearInfo;
    end
end
    
function [outputnodes, infoGain] = childrenAndInfo(inputnode, linSplitThreshold)

m = linSplitThreshold(1);
p = linSplitThreshold(2);

if m == 'X' % Axis aligned x = p
    index = (inputnode(:,1) - p)>0 ;
else % Axis aligned y = p and linear function y = m*x+p
    index = (inputnode(:,2) - p - m*inputnode(:,1))>0 ;
end
outputnodes{1} = inputnode(index == 0,:);
outputnodes{2} = inputnode(index == 1,:);

clear m
clear p
clear index

infoGain = computeInfo(inputnode, outputnodes);

end % return the ouput nodes for desired threhold, and the corresponding info gain

function info = computeInfo(inputnode, outputnodes)
%% Entropy before
for i=1:3
    if ~isempty(inputnode((inputnode(:,3) == i) == 1,:))
        prob(i,1) = length(inputnode((inputnode(:,3) == i) == 1,:))/length(inputnode(:,1));
    end
end
entBefore = sum(-prob.*log(prob),1);
clear prob

%% Entropy After
for j = 1:2
    for i = 1:3
        if ~isempty(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))
            prob(i,j) = length(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))/length(outputnodes{j}(:,1));
        end
    end
end
prob_parent = [length(outputnodes{1}(:,1))/length(inputnode(:,1)) length(outputnodes{2}(:,1))/length(inputnode(:,1))];
entAfter = sum(sum(-prob.*log(prob),1).*prob_parent,2); % Entropy Before
clear prob_parent
clear prob

%% Information gain
info = entBefore - entAfter;
clear entAfter
clear entBefore
end %compute the info gain


%% EDDY'S CODE
% function [childrenBest, infoGainBest] = lineNodeSplit(minGrad, maxGrad, minXInt, maxXInt, rootNode)
%     
%     n = 1;
%     infoGainBest = [0,0,0];
%     %Linear Split Function
%     for i = minGrad:0.1:maxGrad
%         m = 1;
%         for j = minXInt:0.1:maxXInt
%             linSplitThreshold = [i, j];
%             [children, infoGain] = lineSplitFunc(rootNode, linSplitThreshold);
%             if infoGain > infoGainBest(3)
%                  infoGainBest = [m,n,infoGain];
%                  childrenBest = children;
%             end
%             % visNodes(children, replacement, infoGain);
%             m = m + 1;
%         end
%         n = n + 1;
%     end
%     
% end
% 

% function [outputnodes, infogain] = lineSplitFunc(inputnode, linSplitThreshold)
% 
% idx = sign(inputnode(:,2) - linSplitThreshold(1,1)*inputnode(:,1) - linSplitThreshold(1,2)) < 0; % WHAT IS THIS ?
% outputnodes{1} = inputnode(idx,:);
% outputnodes{2} = inputnode(~idx,:);
% 
% %% Entropy before
% for i=1:3
%     if ~isempty(inputnode((inputnode(:,3) == i) == 1,:))
%         prob(i,1) = length(inputnode((inputnode(:,3) == i) == 1,:))/length(inputnode(:,1));
%     end
% end
% entBefore = sum(-prob.*log(prob),1);
% clear prob
% 
% %% Entropy After
% for j = 1:2
%     for i = 1:3
%         if ~isempty(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))
%             prob(i,j) = length(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))/length(outputnodes{j}(:,1));
%         end
%     end
% end
% prob_parent = [length(outputnodes{1}(:,1))/length(inputnode(:,1)) length(outputnodes{2}(:,1))/length(inputnode(:,1))];
% entAfter = sum(sum(-prob.*log(prob),1).*prob_parent,2); % Entropy Before
% 
% %% Information gain
% infogain = entBefore - entAfter
% end

% function visNodes(inputs, replacement, infoGain)
% 
% % Plot the position of the toy present in each bag
% figure
% for i = 1:length(inputs)
%     subplot(2,2,i)
%     for j = 1:length(inputs{i})
%         if inputs{i}(j,3) == 1
%             plot(inputs{i}(j,1),inputs{i}(j,2),'or')
%             hold on
%         elseif inputs{i}(j,3) == 2
%             plot(inputs{i}(j,1),inputs{i}(j,2),'+b')
%             hold on
%         elseif inputs{i}(j,3) == 3
%             plot(inputs{i}(j,1),inputs{i}(j,2),'*g')
%             hold on
%         end
%         if ~isempty(infoGain)
%             if replacement == 0
%                 title({['Bag ' num2str(i) ' without replacement,'];['info gain = ' num2str(infoGain(1,3))]})
%             elseif replacement == 1
%                 title({['Bag ' num2str(i) ' with replacement,'];['info gain = ' num2str(infoGain(1,3))]})
%             end
%         else
%             if replacement == 0
%                 title(['Bag ' num2str(i) ' without replacement'])
%             elseif replacement == 1
%                 title(['Bag ' num2str(i) ' with replacement'])
%             end
%         end
%             xlabel('x co-ordinate')
%             ylabel('y co-ordinate')
%     end
%     grid on
% end