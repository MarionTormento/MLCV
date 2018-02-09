clear all
close all

[data_train, data_test] = getData('Toy_Spiral');

%% Bagging
n = 4; %number of bags, n
infoGain = [];
s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
replacement = 1; % 0 for no replacement and 1 for replacement
[bags] = bagging(n, s, data_train, replacement);
visnodes(bags, replacement, infoGain);

%% Split Function
linSplitThreshold = 0.5;
[children, infoGain] = linsplitfunc(bags{1}, linSplitThreshold);
visnodes(children, replacement, infoGain);

%% Plotting
figure
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

function visnodes(inputs, replacement, infoGain)

figure
for i = 1:length(inputs)
    subplot(2,2,i)
    for ii = 1:length(inputs{i})
        if inputs{i}(ii,3) == 1
            plot(inputs{i}(ii,1),inputs{i}(ii,2),'or')
            hold on
        elseif inputs{i}(ii,3) == 2
            plot(inputs{i}(ii,1),inputs{i}(ii,2),'+b')
            hold on
        elseif inputs{i}(ii,3) == 3
            plot(inputs{i}(ii,1),inputs{i}(ii,2),'*g')
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

figure
for i = 1:length(inputs)
    subplot(2,2,i)
    histogram(inputs{i}(:,3))
    xlabel('Catagory')
    ylabel('# of Occurances')
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

function [outputnodes, infogain] = linsplitfunc(inputnode, boundary)
idx = inputnode(:,1) > boundary;
outputnodes{1}(:,:) = inputnode(idx == 1, :);
outputnodes{2}(:,:) = inputnode(idx == 0, :);

ent1B = length(inputnode((inputnode(:,3) == 1) == 1,:))/length(inputnode(:,1)) * log(length(inputnode((inputnode(:,3) == 1) == 1,:))/length(inputnode(:,1)));
ent2B = length(inputnode((inputnode(:,3) == 2) == 1,:))/length(inputnode(:,1)) * log(length(inputnode((inputnode(:,3) == 2) == 1,:))/length(inputnode(:,1)));
ent3B = length(inputnode((inputnode(:,3) == 3) == 1,:))/length(inputnode(:,1)) * log(length(inputnode((inputnode(:,3) == 3) == 1,:))/length(inputnode(:,1)));
entB = -ent1B -ent2B -ent3B;

for j = 1:2
    for i = 1:3
        if ~isempty(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))
            entA{j}(i) = length(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))/length(outputnodes{j}(:,1)) * log(length(outputnodes{j}((outputnodes{j}(:,3) == i) == 1,:))/length(outputnodes{j}(:,1)));
        end
    end
end
entA1 = -sum(entA{1});
entA2 = -sum(entA{2});

entATotal = length(outputnodes{1}(:,1))/length(inputnode(:,1))*entA1 + length(outputnodes{2}(:,1))/length(inputnode(:,1))*entA2;

infogain = entB - entATotal;
end
