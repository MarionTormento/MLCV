% Separate the training data in param.n bags
function [bags] = bagging(param, data_train)
% choose whether bagging is made with or without replacement and set
% the desired number of bags and bag size
replacement = param.replacement;
n = param.n;
s = param.s;

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