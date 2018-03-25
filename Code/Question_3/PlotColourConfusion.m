clc
close all

load confusionMatrixQ32.mat

figure(1)
pcolor(Conf)
set(gca,'Ydir','reverse')
xlabel('Predicted Class')
ylabel('Actual Class')
title('Confusion Matrix - k-means codebook')
colorbar
caxis([0 100])

load confusionMatrixQ33.mat

figure(2)
pcolor(Conf)
set(gca,'Ydir','reverse')
xlabel('Predicted Class')
ylabel('Actual Class')
title('Confusion Matrix - 5-10-5 RF-codebook configuration')
colorbar
caxis([0 100])