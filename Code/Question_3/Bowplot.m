clear all
clc
close all

load data_train_10.mat
train_data10 = data_train;
load data_train_50.mat
train_data50= data_train;
load data_train_128.mat
train_data128 = data_train;
load data_train_256.mat
train_data256 = data_train;
load data_train_320.mat
train_data320 = data_train;
load data_train_500.mat
train_data500 = data_train;

clear data_train

figure()
subplot(4,1,2)
bar(train_data128(1,:))
title('128 Words')
xlabel('Words')
ylabel('Occurrences')
grid on
subplot(4,1,1)
bar(train_data50(2,:))
title('50 Words')
xlabel('Words')
ylabel('Occurrences')
grid on
subplot(4,1,3)
bar(train_data256(3,:))
title('256 Words')
xlabel('Words')
ylabel('Occurrences')
grid on
subplot(4,1,4)
bar(train_data320(4,:))
title('320 Words')
xlabel('Words')
ylabel('Occurrences')
grid on

comp = [train_data128(81,:);...
        train_data128(82,:);...
        train_data128(83,:)];
figure()
% colormap(colorcube)
bar(comp','grouped')
title('128 Words')
xlabel('Words')
ylabel('Occurrences')
legend('WildCat 1', 'WildCat 2', 'WildCat 3', 'location', 'best')
grid on
axis([0 128 0 max(max(comp))+10])

