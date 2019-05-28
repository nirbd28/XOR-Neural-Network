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
hidden_layer_n=3;%hidden layer nodes number

%%%%%%%%%% run
run_times=100;
for i=1:run_times
    %%%%% init weights
    w1=randn(hidden_layer_n,input_size);%w(i,j): i-hidden layer neuron, j-input
    b1=randn(hidden_layer_n,1);
    w2=randn(output_size,hidden_layer_n);
    b2=randn(output_size,1);
    %%%%%
    [~, ~, ~, ~, cost_arr(i,:)]=Parameters_OPT(features, labels, w1, b1, w2, b2, learning_rate, itter_num);
end
%%% calc mean cost
for i=1:itter_num
   mean_cost(i)=mean(cost_arr(:,i));
end

%%% calc 95 percent
start_value=mean_cost(1);
finish_value=mean_cost(length(mean_cost));
diff_1_perecnt=abs(start_value-finish_value)/100;
diff_95_perecnt=finish_value+diff_1_perecnt*5;

%%% find 95 percent
for i=1:itter_num
    if mean_cost(i) <= diff_95_perecnt
        diff_95_perecnt=mean_cost(i);
        itter_95=i;
        break;
    end
end

%%%%% plot
figure;
plot(mean_cost);
title('cost VS itteration')
xlabel('itter')
ylabel('cost')
%%%
hold on;
scatter(itter_95,mean_cost(itter_95),'*');
legend('Mean cost','95%');

