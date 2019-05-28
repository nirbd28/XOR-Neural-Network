function [w1_opt, b1_opt, w2_opt, b2_opt, cost_arr]=Parameters_OPT(features, labels, w1, b1, w2, b2, learning_rate, itter_num)

%%%%%%%%%% length parameters
features_num=length(labels);
[output_size,~]=size(b2);
[hidden_layer_n,input_size]=size(w1);

%%%%%%%%%% itterations
itter=0;
while 1
    itter=itter+1;
    %%%%% calculate parameters of foward propagation
    [output, struct]=Foward_Propagation_All_Features(features, w1, b1, w2, b2);
    
    %%%%% calculate cost
    cost_arr(itter)=Calc_Cost(labels, output);
    
    %%%%% calculate grad
    [w1_grad, b1_grad]=Grad_Layer1(features, labels, w1, b1, w2, b2, struct);
    [w2_grad, b2_grad]=Grad_Layer2(features, labels, w1, b1, w2, b2, struct);
    
    %%%% update parameters
    w1=w1-w1_grad*learning_rate;
    b1=b1-b1_grad*learning_rate;
    w2=w2-w2_grad*learning_rate;
    b2=b2-b2_grad*learning_rate;
    
    %%%%% break from loop
    if itter==itter_num
       break; 
    end
end

w1_opt=w1;
b1_opt=b1;
w2_opt=w2;
b2_opt=b2;

end