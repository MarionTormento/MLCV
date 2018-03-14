function Acc = accuracy(param, data_test, classpred)
    Acc = 100*sum(classpred(:,1) == data_test(:,param.dimensions+1),1)/size(data_test,1);
end