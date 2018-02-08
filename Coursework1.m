clear all
close all

[data_train, data_test] = getData('Toy_Spiral');

%% Bagging
n = 4; %number of bags, n
s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
replacement = 1; % 0 for no replacement and 1 for replacement
[bags] = bagging(n, s, data_train, replacement);

%% Split Function
idx = bags{1}(:,1) > 0;
child{1}(:,:) = bags{1}(idx == 1, :);
child{2}(:,:) = bags{1}(idx == 0, :);

ent1B = length(bags{1}((bags{1}(:,3) == 1) == 1,:))/length(bags{1}(:,1)) * log(length(bags{1}((bags{1}(:,3) == 1) == 1,:))/length(bags{1}(:,1)));
ent2B = length(bags{1}((bags{1}(:,3) == 2) == 1,:))/length(bags{1}(:,1)) * log(length(bags{1}((bags{1}(:,3) == 2) == 1,:))/length(bags{1}(:,1)));
ent3B = length(bags{1}((bags{1}(:,3) == 3) == 1,:))/length(bags{1}(:,1)) * log(length(bags{1}((bags{1}(:,3) == 3) == 1,:))/length(bags{1}(:,1)));
entB = -ent1B -ent2B -ent3B;

for j = 1:2
    for i = 1:3
        if ~isempty(child{j}((child{j}(:,3) == i) == 1,:))
            entA{j}(i) = length(child{j}((child{j}(:,3) == i) == 1,:))/length(child{j}(:,1)) * log(length(child{j}((child{j}(:,3) == i) == 1,:))/length(child{j}(:,1)));
        end
    end
end
entA1 = -sum(entA{1});
entA2 = -sum(entA{2});

entATotal = length(child{1}(:,1))/length(bags{1}(:,1))*entA1 + length(child{2}(:,1))/length(bags{1}(:,1))*entA2;

InfGain = entB - entATotal;

%% Plotting
figure(1)
scatter(data_test(:,1),data_test(:,2),'.b');

figure(2)
for i = 1:length(data_train)
    if data_train(i,3) == 1
        plot(data_train(i,1),data_train(i,2),'or')
        hold on
    elseif data_train(i,3) == 2
        plot(data_train(i,1),data_train(i,2),'+b')
        hold on
    elseif data_train(i,3) == 3
        plot(data_train(i,1),data_train(i,2),'*g')
        hold on
    end
end
grid on

figure(3)
for i = 1:n
    subplot(2,2,i)
    for ii = 1:length(bags{i})
        if bags{i}(ii,3) == 1
            plot(bags{i}(ii,1),bags{i}(ii,2),'or')
            hold on
        elseif bags{i}(ii,3) == 2
            plot(bags{i}(ii,1),bags{i}(ii,2),'+b')
            hold on
        elseif bags{i}(ii,3) == 3
            plot(bags{i}(ii,1),bags{i}(ii,2),'*g')
            hold on
        end
        if replacement == 0
            title(['Bag ' num2str(i) ' without replacement'])
        elseif replacement == 1
            title(['Bag ' num2str(i) ' with replacement'])
        end
        xlabel('x co-ordinate')
        ylabel('y co-ordinate')
    end
    grid on
end

figure(4)
for i = 1:n
    subplot(2,2,i)
    histogram(bags{i}(:,3))
    xlabel('Catagory')
    ylabel('# of Occurances')
    if replacement == 0
        title(['Bag ' num2str(i) ' without replacement'])
    elseif replacement == 1
        title(['Bag ' num2str(i) ' with replacement'])
    end
    grid on
end

function [bags] = bagging(n, s, data_train, replacement)
    
    if replacement == 1
        %with replacement
        for i = 1:n
            randomIndex = randperm(length(data_train),round(s));
            bags{i}(:,:) = data_train(randomIndex,1:3);
        end
    elseif replacement == 0
        t = round((length(data_train)-1)/n);
        %withoutreplacement
        randomIndexTemp = 1:length(data_train);
        for i = 1:n
            randomIndex = randperm(length(randomIndexTemp),t);
            indx = randomIndexTemp(randomIndex);
            bags{i}(:,:) = data_train(indx,1:3);
            randomIndexTemp(randomIndex) = [];
        end
    end

end
