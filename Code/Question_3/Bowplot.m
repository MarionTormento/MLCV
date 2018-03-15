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

comp = [train_data50(1,:);...
        train_data50(16,:);...
        train_data50(31,:);...
%         train_data50(46,:);...
%         train_data50(61,:);...
%         train_data50(76,:);...
%         train_data50(91,:);...
%         train_data50(106,:);...
%         train_data50(121,:);...
%         train_data50(136,:);...
        ];
    
DiffCat_ANOVA = anova2(comp(:,1:50))

figure()
% colormap(colorcube)
bar(comp','grouped')
title('50 Words')
xlabel('Words')
ylabel('Occurrences')
legend('Tick', 'Trilobite', 'Watch', 'location', 'best')
grid on
axis([0 50 0 max(max(comp))+10])

comp = [train_data50(61,:);...
        train_data50(62,:);...
        train_data50(63,:);...
%         train_data50(64,:);...
%         train_data50(65,:);...
%         train_data50(66,:);...
%         train_data50(67,:);...
%         train_data50(68,:);...
%         train_data50(69,:);...
%         train_data50(70,:);...
        ];
    
sameCat_ANOVA = anova2(comp(:,1:50))

figure()
% colormap(colorcube)
bar(comp','grouped')
title('50 Words')
xlabel('Words')
ylabel('Occurrences')
legend('Watch 1', 'Watch 2', 'Watch 3', 'location', 'best')
grid on
axis([0 50 0 max(max(comp))+10])

