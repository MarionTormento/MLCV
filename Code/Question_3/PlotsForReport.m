clear all
close all

load Accuracy_5_10_5_2.mat

figure
Trees = [10 15 30 50];
for i = 1:4
    subplot(2,2,i)
    idx = AccTot(:,1) == Trees(i);
    colormap = varycolor(sum(idx));
    for ii = 1:sum(idx)
        idy = AccTot(:,3) == ii*5;
        plot(AccTot(idx&idy,2),AccTot(idx&idy,4), 'color', colormap(ii,:), 'marker', 'o', 'linestyle', '-')
        hold on
    end
    title([num2str(Trees(i)) ' Trees'])
    xlabel('Number of layers')
    ylabel('Accuracy %')
    grid on
    axis([7 9 40 70])
end
    legend('\rho = 5','\rho = 10')

