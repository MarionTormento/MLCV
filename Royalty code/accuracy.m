function Acc = accuracy(param, data_train, leaves, nodes)
    [classpred] = testForest(param, data_train, leaves, nodes, 0, 0);
    Acc = 100*sum(classpred(:,1) == data_train(:,3),1)/size(data_train,1);
end