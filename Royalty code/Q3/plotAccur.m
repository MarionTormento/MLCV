load('Accuracy_randomfull.mat')
k=1;

for i=[5 10 15]
    Fct{k} = []
    for j = 1:length(AccTot)
        if AccTot(j,3) == i;
            Fct{k} = [Fct{k}; AccTot(j,:)];
        end
    end
    k = k+1;
end

hFig = figure();
set(gcf, 'Color', 'White')
axh = axes('Parent', hFig);
hold(axh, 'all');
h1 = scatter3(Fct{1}(:,1),Fct{1}(:,2),Fct{1}(:,4));
h2 = scatter3(Fct{2}(:,1),Fct{2}(:,2),Fct{2}(:,4));
h3 = scatter3(Fct{3}(:,1),Fct{3}(:,2),Fct{3}(:,4));
view(axh, -33, 22);
grid(axh, 'on');
legend(axh, [h1, h2, h3], {"NumSplit = 5","NumSplit = 10","NumSplit = 15"});
xlabel('Number of trees');
ylabel('Number of layers');
zlabel('Accuracy (%)');