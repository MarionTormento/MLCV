clear all
close all
clc

Vocabulary = [10, 50, 128, 256, 320, 500];
% Accuracy = [58, 58.67, 55.33, 64.67, 63.33, 58.67];
% time = [25.6, 27.92, 29.26, 38.83, 39.56, 49.07];
Accuracy = [56.67, 58.67, 58.3, 60.67, 65.33, 48.67];
time = [14.45, 15.46, 15.72, 17.83, 20.56, 25.07];

figure()
plot(Vocabulary,Accuracy);
hold on
plot(Vocabulary,time);
xlabel('Vocabulary size')
legend('Accuracy (%)', 'Computational time (s)')