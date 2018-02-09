clear all
close all
clc

[data_train, data_test] = getData('Toy_Spiral');

%% Bagging

n = 4; %number of bags, n
s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
replacement = 1; % 0 for no replacement and 1 for replacement

infoGain = []; %initialise infoGain

% bagging and visualise bags, Choose a bag for root node.
[bags] = bagging(n, s, data_train, replacement);
visNodes(bags, replacement, infoGain);

%% Split Function
%[children, infoGain] = lineNodeSplit(-3, 3, -0.5, 0.5, rootNode);
[children, infoGain] = lineNodeSplit(-1, 1, -1, 1, rootNode);


%% Recursive test
% % Branch 0 -> 1 & 2
% rootNode = bags{1};
% [children, infoGain] = lineNodeSplit(-1,1,-1,1,rootNode);
% clear rootNode
% visNodes(children, replacement, infoGain);
% 
% % Branch 2 -> 21 & 22
% rootNode = children{2};
% [john, infoGain] = lineNodeSplit(-1,1,-1,1,rootNode);
% clear rootNode
% visNodes(john, replacement, infoGain);
% 
% % Branch 21 -> 211 & 212
% rootNode = john{1};
% [bob, infoGain] = lineNodeSplit(-1,1,-1,1,rootNode);
% clear rootNode
% visNodes(bob, replacement, infoGain);

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

function visNodes(inputs, replacement, infoGain)

% Plot the position of the toy present in each bag
figure
for i = 1:length(inputs)
    subplot(2,2,i)
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
        if ~isempty(infoGain)
            if replacement == 0
                title({['Bag ' num2str(i) ' without replacement,'];['info gain =' num2str(infoGain)]})
            elseif replacement == 1
                title({['Bag ' num2str(i) ' with replacement,'];['info gain =' num2str(infoGain)]})
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

% Plot the histogram of the toy class repartition in each bag
figure
for i = 1:length(inputs)
    subplot(2,2,i)
    histogram(inputs{i}(:,3))
    xlabel('Category')
    ylabel('# of Occurences')
    if ~isempty(infoGain)
        if replacement == 0
            title({['Bag ' num2str(i) ' without replacement,'];['info gain =' num2str(infoGain)]})
        elseif replacement == 1
            title({['Bag ' num2str(i) ' with replacement,'];['info gain =' num2str(infoGain)]})
        end
    else
        if replacement == 0
            title(['Bag ' num2str(i) ' without replacement'])
        elseif replacement == 1
            title(['Bag ' num2str(i) ' with replacement'])
        end
    end
    grid on
end

end

%% MARION CODE
function [childrenBest, infoGainBest] = lineNodeSplit(minX, maxX, minY, maxY, rootNode)
    
    infoGainBest = [0,0,0];
    %Linear Split Function for x=i
    for i = minX:0.1:maxX
        linSplitThreshold = [1, i];
        [children, infoGain] = lineSplitFunc(rootNode, linSplitThreshold);
        if infoGain > infoGainBest(3)
            infoGainBest = [1, i, infoGain];
            childrenBest = children;
        end
    end
    %Linear Split Function for y=i
    for i = minY:0.1:maxY
        linSplitThreshold = [2, i];
        [children, infoGain] = lineSplitFunc(rootNode, linSplitThreshold);
        if infoGain > infoGainBest(3)
            infoGainBest = [2,i,infoGain];
            childrenBest = children;
        end
    end
    clear children
    clear infoGain
    clear linSplitThreshold
end
    
function [outputnodes, infogain] = lineSplitFunc(inputnode, linSplitThreshold)

split_idx = linSplitThreshold(1);
split_th = linSplitThreshold(2);
index = inputnode(:,split_idx) > split_th;
outputnodes{1} = inputnode(index == 0,:);
outputnodes{2} = inputnode(index == 1,:);

clear split_idx
clear split_th
clear index

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
infogain = entBefore - entAfter;
clear entAfter
clear entBefore
end


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
% function [outputnodes, infogain] = axisSplitFunc(inputnode, boundary) 
% idx = inputnode(:,1) > boundary;
% outputnodes{1} = inputnode(idx == 1, :);
% outputnodes{2} = inputnode(idx == 0, :);
% 
% ent1B = length(inputnode((inputnode(:,3) == 1) == 1,:))/length(inputnode(:,1)) * log(length(inputnode((inputnode(:,3) == 1) == 1,:))/length(inputnode(:,1)));
% ent2B = length(inputnode((inputnode(:,3) == 2) == 1,:))/length(inputnode(:,1)) * log(length(inputnode((inputnode(:,3) == 2) == 1,:))/length(inputnode(:,1)));
% ent3B = length(inputnode((inputnode(:,3) == 3) == 1,:))/length(inputnode(:,1)) * log(length(inputnode((inputnode(:,3) == 3) == 1,:))/length(inputnode(:,1)));
% entB = -ent1B -ent2B -ent3B;
% 
% for j = 1:2
%     for i = 1:3
%         if ~isempty(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))
%             entA{j}(i) = length(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))/length(outputnodes{j}(:,1)) * log(length(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))/length(outputnodes{j}(:,1)));
%         end
%     end
% end
% entA1 = -sum(entA{1});
% entA2 = -sum(entA{2});
% 
% entATotal = length(outputnodes{1}(:,1))/length(inputnode(:,1))*entA1 + length(outputnodes{2}(:,1))/length(inputnode(:,1))*entA2;
% 
% infogain = entB - entATotal;
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
% 
% %% BACK UP PREVIOUS CODE
% % ent1B = length(inputnode((inputnode(:,3) == 1) == 1,:))/length(inputnode(:,1)) * log(length(inputnode((inputnode(:,3) == 1) == 1,:))/length(inputnode(:,1)));
% % ent2B = length(inputnode((inputnode(:,3) == 2) == 1,:))/length(inputnode(:,1)) * log(length(inputnode((inputnode(:,3) == 2) == 1,:))/length(inputnode(:,1)));
% % ent3B = length(inputnode((inputnode(:,3) == 3) == 1,:))/length(inputnode(:,1)) * log(length(inputnode((inputnode(:,3) == 3) == 1,:))/length(inputnode(:,1)));
% % entB2 = -ent1B -ent2B -ent3B;
% % 
% % for j = 1:2
% %     for i = 1:3
% %         if ~isempty(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))
% %             entA{j}(i) = length(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))/length(outputnodes{j}(:,1)) * log(length(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))/length(outputnodes{j}(:,1)));
% %         end
% %     end
% % end
% % entA1 = -sum(entA{1});
% % entA2 = -sum(entA{2});
% % entATotal = length(outputnodes{1}(:,1))/length(inputnode(:,1))*entA1 + length(outputnodes{2}(:,1))/length(inputnode(:,1))*entA2;
% end