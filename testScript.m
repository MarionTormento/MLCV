clear all
close all

[data_train, data_test] = getData('Toy_Spiral');

%% Bagging

n = 4; %number of bags, n
s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
replacement = 1; % 0 for no replacement and 1 for replacement
leafCount = 1;

infoGain = []; %initialise infoGain

% bagging and visualise bags, Choose a bag for root node.
[bags] = bagging(n, s, data_train, replacement);
visBags(bags, replacement, infoGain);

%% Split Function
param.numfunct = 3;
param.numlevels = 6;
param.rho = 0.8;

%% Recursive test

for k = 1:n
    
    %initialize
    tree{1,k} = cell(param.numlevels, 2^(param.numlevels-1));
    
    %Split the root node into the initial children
    rootNode = bags{k};
    tree{1,k}{1,1} = rootNode;
    [children, infoGain] = optimalNodeSplit(param, rootNode);
    clear rootNode
    visNodes(children, replacement, infoGain, k, 1);
    clear infoGain
    tree{1,k}{2,1} = children{1};
    tree{1,k}{2,2} = children{2};

    %number of levels in the tree
    for j = 3:param.numlevels 
        %for each child we decide on an optimum split function
        %idx = length(children)
        for i = 1:(2^(j-2))
            if isempty(children{i})
                childrenNext{2*i-1} = [];
                childrenNext{2*i} = [];
                continue
            end
            rootNode = children{i};
            [childrenNew, infoGain] = optimalNodeSplit(param, rootNode);
            visNodes(childrenNew, replacement, infoGain, k, j);
            for ii = 1:length(childrenNew)
                isLeaf = leafTest(childrenNew{ii}, infoGain); 
                if isLeaf
                    prob1 = sum(childrenNew{ii}(:,3) == 1)/(sum(childrenNew{ii}(:,3) == 1)+sum(childrenNew{ii}(:,3) == 2)+sum(childrenNew{ii}(:,3) == 3));
                    prob2 = sum(childrenNew{ii}(:,3) == 2)/(sum(childrenNew{ii}(:,3) == 1)+sum(childrenNew{ii}(:,3) == 2)+sum(childrenNew{ii}(:,3) == 3));
                    prob3 = sum(childrenNew{ii}(:,3) == 3)/(sum(childrenNew{ii}(:,3) == 1)+sum(childrenNew{ii}(:,3) == 2)+sum(childrenNew{ii}(:,3) == 3));
                    leaf{leafCount} = [prob1, prob2, prob3];
                    leafCount = leafCount + 1;
                    if ii == 1
                        childrenNext{2*i-1} = [];
                    elseif ii == 2
                        childrenNext{2*i} = []; 
                    end
                end
                if ~(isLeaf)
                    tree{1,k}{j,2*i-1} = childrenNew{1};  
                    tree{1,k}{j,2*i} = childrenNew{2};
                    % Collect a next children array for next branch
                    if ii == 1
                        childrenNext{2*i-1} = childrenNew{1};
                    elseif ii == 2
                        childrenNext{2*i} = childrenNew{2};
                    end
                end
            end
            % Complete the tree

            clear rootNode
            clear childrenNew
            clear infoGain
           % pause               
        end
        
        clear children
        %redefine our new generation of children as our current children for
        %the next layer of the tree
        children = childrenNext;
        clear ChildrenNext
    end
    clear children
end


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

function visBags(inputs, replacement, infoGain)
% Plot the position of the toy present in each bag
figure(1)
for i = 1:length(inputs)
    subplot(2,2,i)
    for j = 1:size(inputs{i},1)
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
        if ~isempty(infoGain)
            if replacement == 0
                title({['Bag ' num2str(i) ' without replacement,'];['info gain = ' num2str(infoGain(1,3))]})
            elseif replacement == 1
                title({['Bag ' num2str(i) ' with replacement,'];['info gain = ' num2str(infoGain(1,3))]})
            end
        else
            if replacement == 0
                title(['Bag ' num2str(i) ' without replacement'])
            elseif replacement == 1
                title(['Bag ' num2str(i) ' with replacement'])
            end
        end
            xlabel('x co-ordinate')
            ylabel('y co-ordinate')
    end
    grid on
end
end

function visNodes(inputs, replacement, infoGain, k, jj)

    %error debugging, ensuring the input cell is a structure if one of the children in empty
    if ~iscell(inputs)
        inputscell{1}(:,:) = inputs;
        inputscell{2} =[];
        clear children
        inputs = inputscell;
    end

    % Plot the position of the toy present in each bag
    figure(2)

    if infoGain.x1 == 'X'
       threshold_y = -1:0.1:1;
       threshold_x = infoGain.x2*ones(1,length(threshold_y));
    elseif infoGain.x1 == 'Y'
       threshold_x = -1:0.1:1;
       threshold_y = infoGain.x2*ones(1,length(threshold_x));
    else
       threshold_x = -1:0.1:1; 
       threshold_y = infoGain.x1.*threshold_x+infoGain.x2;
    end

    for i = 1:length(inputs)
        subplot(2,2,1)
        for j = 1:size(inputs{i},1)
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
                title({['Parent and threshold without replacement,'];['info gain = ' num2str(infoGain.Gain)]})
            elseif replacement == 1
                title({['Parent and threshold with replacement,'];['info gain = ' num2str(infoGain.Gain)]})
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
    if ~isempty(infoGain)
            text(2,0.5,{['Tree Number = ' num2str(k)],['Tree Level = ' num2str(jj)],...
                ['Best info gain = ' num2str(infoGain.Gain)]})
    else
            text(2,0.5,{['Tree Number = ' num2str(k)],['Tree Level = ' num2str(jj)]})
    end
    hold off
    
    % Plot the histogram of the toy class repartition in each bag
    %figure
    for i = 1:length(inputs)
        subplot(2,2,i+2)
        if ~isempty(inputs{i}) %if there are no points in the child, don't plot histogram or errors
            histogram(inputs{i}(:,3), 0.5:1:3.5)
        end
        xlabel('Category')
        ylabel('# of Occurences')
        title(['Child ' num2str(i) ','])
        grid on
        hold off
    end

end

function [childrenBest, infoGainBest] = axisNodeSplit(minX, maxX, minY, maxY, rootNode, numfunct) % Compute the best 'x=...' split node for the bag
    
    infoGainBest.x1 = 'X';
    infoGainBest.x2 = 0;
    infoGainBest.Gain = 0;
    childrenBest = [];
    
    randomSampX = randperm(round((maxX-minX)/0.001),numfunct);
    % Axis Split Function for x=i
    linSplitThreshold.x1 = 'X';
    for i = 1:numfunct
        threshold = minX + 0.001*randomSampX(i);
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
    randomSampY = randperm(round((maxY-minY)/0.001),numfunct);
    for i = 1:numfunct
        threshold = minY + 0.001*randomSampY(i);
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

function [childrenBest, infoGainBest] = linearNodeSplit(minYInt, maxYInt, rootNode, numfunct) % Compute the best "y=mx+p" split node for the bag
    
    infoGainBest.x1 = 0;
    infoGainBest.x2 = 0;
    infoGainBest.Gain = 0;
    childrenBest = [];
    randomSampYInt = randperm(round((abs(maxYInt - minYInt))/0.01),numfunct);
    %given an y intercept, calculate good max and min gradients
    if rootNode(rootNode(:,2) == min(rootNode(:,2)),1) < 0
        maxGrad = (randomSampYInt - min(rootNode(:,2))/0.01)*0.01./(-rootNode(rootNode(:,2) == min(rootNode(:,2)),1));
        minGrad = (randomSampYInt - max(rootNode(:,2))/0.01)*0.01./(-rootNode(rootNode(:,2) == max(rootNode(:,2)),1));
    else 
        minGrad = (randomSampYInt - min(rootNode(:,2))/0.01)*0.01./(-rootNode(rootNode(:,2) == min(rootNode(:,2)),1));
        maxGrad = (randomSampYInt - max(rootNode(:,2))/0.01)*0.01./(-rootNode(rootNode(:,2) == max(rootNode(:,2)),1));
    end
    %Linear Split Function y = m*x+p
    for m = 1:numfunct
        %for each yint, calc 3 random grads between that points man and min
        %grad
        randomSampGrad(m,:) = minGrad/0.01 + randperm(round((abs(maxGrad(m)+minGrad(m))/0.01)),numfunct);
        for p = 1:numfunct
            linSplitThreshold.x1 = randomSampGrad(m,p)*0.01;
            linSplitThreshold.x2 = randomSampYInt(p)*0.01;
            [children, infoGain] = childrenAndInfo(rootNode, linSplitThreshold);
            if infoGain > infoGainBest.Gain
                 infoGainBest.x1 = randomSampGrad(m,p)*0.01;
                 infoGainBest.x2 = randomSampYInt(p)*0.01;
                 infoGainBest.Gain = infoGain;
                 childrenBest = children;
            end
         end
    end
end

function [childrenBest, infoGainBest] = optimalNodeSplit(param, rootNode) % compute the optimal split node between axis and linear
    
    rho = round(length(rootNode)*param.rho);
    numfunct = param.numfunct;
    
    %data = rootNode(randperm(length(rootNode),rho),:);
        
    X = [min(rootNode(:,1)), max(rootNode(:,1))];
    YInt = [min(rootNode(:,2)), max(rootNode(:,2))];
    
    [axisCh, axisInfo] = axisNodeSplit(X(1), X(2), YInt(1), YInt(2), rootNode, numfunct);
    [linearCh, linearInfo] = linearNodeSplit(YInt(1), YInt(2), rootNode, numfunct);
    
    [maxInfo idxInfo] = max([axisInfo.Gain, linearInfo.Gain]); %if idxInfo return 1 => Axis, 2 => linear
    if idxInfo == 1 
            childrenBest = axisCh;
            infoGainBest = axisInfo;
    elseif idxInfo == 2
            childrenBest = linearCh;
            infoGainBest = linearInfo;
    end
end
    
function [outputnodes, infoGain] = childrenAndInfo(inputnode, linSplitThreshold)

m = linSplitThreshold.x1;
p = linSplitThreshold.x2;

if m == 'X' % Axis aligned x = p
    index = (inputnode(:,1) - p)>0 ;
elseif m == 'Y' %axis aligned y = p
    index = (inputnode(:,2) - p)>0 ; 
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
        prob(i,1) = size(inputnode((inputnode(:,3) == i) == 1,:),1)/size(inputnode(:,1),1);
    else
        prob(i,1) = 1; 
    end
end
entBefore = sum(-prob.*log(prob),1);
clear prob

%% Entropy After
for j = 1:2
    for i = 1:3
        if ~isempty(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))
            prob(i,j) = size(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:),1)/size(outputnodes{j}(:,1),1);
        else
           prob(i,j) = 1; 
        end
    end
end
prob_parent = [size(outputnodes{1}(:,1),1)/size(inputnode(:,1),1) size(outputnodes{2}(:,1),1)/size(inputnode(:,1),1)];
entAfter = sum(sum(-prob.*log(prob),1).*prob_parent,2); % Entropy Before
clear prob_parent
clear prob

%% Information gain
info = entBefore - entAfter;
clear entAfter
clear entBefore
end %compute the info gai

function binCount = leafTest(rootNode, infoGain)
    for i = 1:3
        bin(i,1) = isempty(rootNode((rootNode(:,3) == i) == 1,:));
    end
    if sum(bin,1) < 2 && (length(rootNode) > 5) && (infoGain.Gain < 60)
        binCount = 0;
    else
        binCount = 1;
    end
end