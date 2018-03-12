figure
subplot(1,2,1)
for j = 1:size(rootNode,1)
            if rootNode(j,3) == 1
                plot(rootNode(j,1),rootNode(j,2),'or')
                hold on
            elseif rootNode(j,3) == 2
                plot(rootNode(j,1),rootNode(j,2),'+b')
                hold on
            elseif rootNode(j,3) == 3
                plot(rootNode(j,1),rootNode(j,2),'*g')
                hold on
            end
end
grid on
xlabel('x-coordinate')
ylabel('y-coordinate')
title('axis-aligned splits')

threshold_y = -1:0.1:1;
threshold_x = splitPoss.Xthr{1}*ones(1,length(threshold_y));
plot(threshold_x,threshold_y, 'b')
threshold_x = splitPoss.Xthr{2}*ones(1,length(threshold_y));
plot(threshold_x,threshold_y, 'b')

threshold_x = -1:0.1:1;
threshold_y = splitPoss.Ythr{1}*ones(1,length(threshold_x));
plot(threshold_x,threshold_y, 'r')
threshold_y = splitPoss.Ythr{2}*ones(1,length(threshold_x));
plot(threshold_x,threshold_y, 'r')

subplot(1,2,2)
for j = 1:size(rootNode,1)
            if rootNode(j,3) == 1
                plot(rootNode(j,1),rootNode(j,2),'or')
                hold on
            elseif rootNode(j,3) == 2
                plot(rootNode(j,1),rootNode(j,2),'+b')
                hold on
            elseif rootNode(j,3) == 3
                plot(rootNode(j,1),rootNode(j,2),'*g')
                hold on
            end
end
grid on
xlabel('x-coordinate')
ylabel('y-coordinate')
title('linear splits')
threshold_x = -1:0.1:1; 
threshold_y = splitPoss.Lthr{1}(1,1).*threshold_x+splitPoss.Lthr{1}(1,2);
plot(threshold_x,threshold_y, 'b')
threshold_y = splitPoss.Lthr{2}(1,1).*threshold_x+splitPoss.Lthr{2}(1,2);
plot(threshold_x,threshold_y, 'b')
threshold_y = splitPoss.Lthr{3}(1,1).*threshold_x+splitPoss.Lthr{3}(1,2);
plot(threshold_x,threshold_y, 'b')
threshold_y = splitPoss.Lthr{4}(1,1).*threshold_x+splitPoss.Lthr{4}(1,2);
plot(threshold_x,threshold_y, 'b')
axis([-1 1 -1 1])
