% Probability of each class in the leaf
function leaf = leafProb(rootNode, leaf, i, j, k)
%Calculate the probabilites for each classification
prob1 = sum(rootNode(:,3) == 1)/size(rootNode,1);
prob2 = sum(rootNode(:,3) == 2)/size(rootNode,1);
prob3 = sum(rootNode(:,3) == 3)/size(rootNode,1);
%Create the leaf and tag it with its tree num,
newLeaf = [k, j, i, prob1, prob2, prob3];
leaf = [leaf; newLeaf];
end