% *Nir Ben Dor (305136608), Avishai Weizman (315027318)*
%%
clc; clear all; close all;
%%
clc; clear all; close all;

addpath(strcat(pwd,'\','Functions_NN'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% load data
load('features.mat')
load('labels.mat')
[input_size,features_num]=size(features);
output_size=1;

%%%%%%%%%% network inputs
itter_num=2000;
learning_rate=2;
hidden_layer_n=3;

%%%%% init weights
w1=randn(hidden_layer_n,input_size);%w(i,j): i-hidden layer neuron, j-input
b1=randn(hidden_layer_n,1);
w2=randn(output_size,hidden_layer_n);
b2=randn(output_size,1);
%%%%%
[w1, b1, w2, b2, cost_arr]=Parameters_OPT(features, labels, w1, b1, w2, b2, learning_rate, itter_num);

%%%%%%%%%% plot cost
figure;
plot(cost_arr)
title('cost VS itteration')
xlabel('itter')
ylabel('cost')

%%%%%%%%%% check
succes_rate=Check_NN(features, labels, w1, b1, w2, b2);


