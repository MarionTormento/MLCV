% train the RF
function [leaf, splitFct] = trainForest3(bags, param)
    
    %initialise tree structure
    leaf = [];

    %for each tree
    for k = 1:param.n
        
        %initialize the stucture storing the split function parameters
        splitFct{1,k} = cell(param.numlevels,2^(param.numlevels-1));
        
        %Split the root node into the initial children
        rootNode = bags{k};
        %Find optimal split function which maximises the information gain
        [children, infoGain] = optimalNodeSplit3(param, rootNode);
        splitFct{1,k}{1,1} = infoGain;
        
        %Prepare next generation and clear some variables
        parent = children;
        clear rootNode
        clear infoGain
        clear children
        
        %Populating the rest of the tree
        for j = 2:param.numlevels
            % For each parent of the layer
            for i = 1:length(parent)
                rootNode = parent{i};
                
                % If the parent is empty set its children to empty and
                % continue
                if isempty(rootNode)
                    children{2*i-1} = cell(0);
                    children{2*i} = cell(0);
                    continue
                end
                
                %If the root is already a leaf make its children empty and continue
                isRootLeaf = leafTest3(param, rootNode);
                if isRootLeaf
                    children{2*i-1} = cell(0);
                    children{2*i} = cell(0);
                    leaf = leafProb3(param, rootNode, leaf, i, j, k);
                    splitFct{1,k}{j,i}.x1 = 'Leaf';
                    continue
                end
                
                %If we are on the last layer, we don't even try to split
                % the parent and set every splitFct to "Leaf"
                if j == param.numlevels
                    leaf = leafProb3(param, rootNode, leaf, i, j, k);
                    splitFct{1,k}{j,i}.x1 = 'Leaf';
                    continue
                end
                
                %Therefore, the root is not a leaf or empty, perform split function
                [childrenNew, infoGain] = optimalNodeSplit3(param, rootNode);
                splitFct{1,k}{j,i} = infoGain;
                children{2*i-1} = childrenNew{1};
                children{2*i} = childrenNew{2};
                clear rootNode
                clear childrenNew
                clear infoGain
            end
            
            %redefine our new generation of children as our current children for
            %the next layer of the tree
            clear parent
            parent = children;
            clear Children
            disp('...')
        end
        clear parent
    end
    
end
