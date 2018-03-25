nbLeaf = 12;
max = 3;
trees = sort(randperm(param.n,4));
for i=1:max
    idx = sort(randperm(nbLeaf,max));
    data = leaves((leaves(:,1) == trees(i))==1,4:6);
    for j = 1:max
        subplot(max, max, (i-1)*max+j)
        bar(data(idx(j),:));
        title(['Leaf ', num2str(idx(j)), ' from Tree ', num2str(trees(i))]);
    end
    clear data;
end
