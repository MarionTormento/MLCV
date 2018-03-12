load('accur1.mat')
k=1;

for i=3:10
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
h1 = surf(Fct{1}(:,1),Fct{1}(:,2),Fct{1}(:,4));
h2 = surf(Fct{2}(:,1),Fct{2}(:,2),Fct{2}(:,4));
h3 = surf(Fct{3}(:,1),Fct{3}(:,2),Fct{3}(:,4));
h4 = surf(Fct{4}(:,1),Fct{4}(:,2),Fct{4}(:,4));
h5 = surf(Fct{5}(:,1),Fct{5}(:,2),Fct{5}(:,4));
h6 = surf(Fct{6}(:,1),Fct{6}(:,2),Fct{6}(:,4));
h7 = surf(Fct{7}(:,1),Fct{7}(:,2),Fct{7}(:,4));
h8 = surf(Fct{8}(:,1),Fct{8}(:,2),Fct{8}(:,4));
view(axh, -33, 22);
grid(axh, 'on');
legend(axh, [h1, h2, h3], {"NumSplit = 3","NumSplit = 4","NumSplit = 5","NumSplit = 6","NumSplit = 7","NumSplit = 8","NumSplit = 9","NumSplit = 10"});
xlabel('Number of trees');
ylabel('Number of layers');
zlabel('Accuracy (%)');