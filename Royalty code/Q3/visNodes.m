function visNodes(inputs, infoGain, k, jj)

    %error debugging, ensuring the input cell is a structure if one of the children in empty
    if ~iscell(inputs)
        inputscell{1}(:,:) = inputs;
        inputscell{2} =[];
        clear children
        inputs = inputscell;
    end

    % Plot the position of the toy present in each bag
    figure(2)

    if infoGain.x1 == 'X'
       threshold_y = -1:0.1:1;
       threshold_x = infoGain.x2*ones(1,length(threshold_y));
    elseif infoGain.x1 == 'Y'
       threshold_x = -1:0.1:1;
       threshold_y = infoGain.x2*ones(1,length(threshold_x));
    else
       threshold_x = -1:0.1:1; 
       threshold_y = infoGain.x1.*threshold_x+infoGain.x2;
    end

    for i = 1:length(inputs)
        subplot(2,2,1)
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
        end
        title({['Parent and split function']})
        xlabel('x co-ordinate')
        ylabel('y co-ordinate')
        plot(threshold_x,threshold_y)
        axis([-1 1 -1 1])
        grid on
    end
    if ~isempty(infoGain)
            text(2,0.5,{['Tree Number = ' num2str(k)],['Tree Level = ' num2str(jj)],...
                ['Best info gain = ' num2str(infoGain.Gain)]})
    else
            text(2,0.5,{['Tree Number = ' num2str(k)],['Tree Level = ' num2str(jj)]})
    end
    hold off
    
    % Plot the histogram of the toy class repartition in each bag
    %figure
    for i = 1:length(inputs)
        subplot(2,2,i+2)
        if ~isempty(inputs{i}) %if there are no points in the child, don't plot histogram or errors
            histogram(inputs{i}(:,3), 0.5:1:3.5)
        end
        xlabel('Category')
        ylabel('# of Occurences')
        title(['Child ' num2str(i) ','])
        grid on
        hold off
    end

end
