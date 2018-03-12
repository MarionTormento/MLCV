function leaf = leafProb(param, rootNode, leaf, i, j, k)
for m = 1:10
%Calculate the probabilites for each classification
    prob(1,m) = sum(rootNode(:,param.dimensions+1) == m)/size(rootNode,1);
end
%Create the leaf and tag it with its tree num,
newLeaf = [k, j, i, prob];
leaf = [leaf; newLeaf];

end