function visBags(inputs, replacement)
% Plot the position of the toy present in each bag
figure(1)
for i = 1:length(inputs)
    subplot(2,2,i)
    for j = 1:size(inputs{i},1)
        if inputs{i}(j,3) == 1
            plot(inputs{i}(j,1),inputs{i}(j,2),'or')
            hold on
        elseif inputs{i}(j,3) == 2
            plot(inputs{i}(j,1),inputs{i}(j,2),'+b')
            hold on
        elseif inputs{i}(j,3) == 3
            plot(inputs{i}(j,1),inputs{i}(j,2),'*g')
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
end