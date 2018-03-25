clear all
close all

load Accuracy_randomfull2.mat

figure
for i = 1:4
    subplot(2,2,i)
    idx = AccTot(:,1) == 10+(i-1)*5;
    colormap = varycolor(sum(idx));
    for ii = 1:sum(idx)
        idy = AccTot(:,3) == ii*5;
        plot(AccTot(idx&idy,2),AccTot(idx&idy,4), 'color', colormap(ii,:), 'marker', 'o', 'linestyle', '-')
        hold on
    end
    title([num2str((i-1)*5+10) ' Trees'])
    xlabel('Number of layers')
    ylabel('Accuracy %')
    grid on
    axis([7 9 45 70])
end
    legend('\rho = 5','\rho = 10','\rho = 15')
