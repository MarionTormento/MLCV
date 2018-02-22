clear all
close all
clc

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
    parent = children;
    clear children;

    %number of levels in the tree
    for j = 3:param.numlevels
        % For each child
        for i = 1:(2^(j-2))
            % If the child is empty set its children to empty and
            % continue
            if isempty(parent{i})
                childrenNext{2*i-1} = [];
                childrenNext{2*i} = [];
                continue
            end
            %If it's the final layer, don't bother splitting nodes or 
            %making children, just make a leaf and continue
            rootNode = parent{i};
            %Test if the root will have previously been deemed a leaf node
            isRootLeaf = leafTest(rootNode);
            %If the root is already a leaf make its children empty and continue
            if isRootLeaf
                childrenNext{2*i-1} = [];
                childrenNext{2*i} = [];
                continue
            end
            %Elseif the root is not a leaf, perform split function
            [childrenNew, infoGain] = optimalNodeSplit(param, rootNode);
            visNodes(childrenNew, replacement, infoGain, k, j);
            for ii = 1:length(childrenNew)
                % For each childNew, check if it is a leaf
                isChildLeaf = leafTest(childrenNew{ii}); 
                % If the child is a leaf or it's the last layer of the tree,
                % create a leaf node
                if isChildLeaf
                    %Calculate the probabilites for each classification
                    prob1 = sum(childrenNew{ii}(:,3) == 1)/(sum(childrenNew{ii}(:,3) == 1)+sum(childrenNew{ii}(:,3) == 2)+sum(childrenNew{ii}(:,3) == 3));
                    prob2 = sum(childrenNew{ii}(:,3) == 2)/(sum(childrenNew{ii}(:,3) == 1)+sum(childrenNew{ii}(:,3) == 2)+sum(childrenNew{ii}(:,3) == 3));
                    prob3 = sum(childrenNew{ii}(:,3) == 3)/(sum(childrenNew{ii}(:,3) == 1)+sum(childrenNew{ii}(:,3) == 2)+sum(childrenNew{ii}(:,3) == 3));
                    % Create the leaf and tag it with its tree num,
                    % layer num etc.
                    leaf{leafCount} = [k, j, 2*i-(2-ii), prob1, prob2, prob3];
                    leafCount = leafCount + 1;
                    tree{1,k}{j,2*i-(2-ii)} = childrenNew{ii};
                    childrenNext{2*i-(2-ii)} = childrenNew{ii}; 
                %Elseif child is not leaf, create its new children
                elseif ~(isChildLeaf)
                   tree{1,k}{j,2*i-(2-ii)} = childrenNew{ii};
                   childrenNext{2*i-(2-ii)} = childrenNew{ii};
                end
                if j == param.numlevels
                    %Calculate the probabilites for each classification
                    prob1 = sum(childrenNew{ii}(:,3) == 1)/(sum(childrenNew{ii}(:,3) == 1)+sum(childrenNew{ii}(:,3) == 2)+sum(childrenNew{ii}(:,3) == 3));
                    prob2 = sum(childrenNew{ii}(:,3) == 2)/(sum(childrenNew{ii}(:,3) == 1)+sum(childrenNew{ii}(:,3) == 2)+sum(childrenNew{ii}(:,3) == 3));
                    prob3 = sum(childrenNew{ii}(:,3) == 3)/(sum(childrenNew{ii}(:,3) == 1)+sum(childrenNew{ii}(:,3) == 2)+sum(childrenNew{ii}(:,3) == 3));
                    %Create the leaf and tag it with its tree num,
                    %layer num etc.
                    leaf{leafCount} = [k, j, 2*i-(2-ii), prob1, prob2, prob3];
                    leafCount = leafCount + 1;
                end
            end
            % Complete the tree

            clear rootNode
            clear childrenNew
            clear infoGain
           % pause               
        end
        
        clear parent
        %redefine our new generation of children as our current children for
        %the next layer of the tree
        parent = childrenNext;
        clear ChildrenNext
    end
    clear parent
end
