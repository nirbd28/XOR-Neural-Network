function [w1_grad, b1_grad]=Grad_Layer1(features, labels, w1, b1, w2, b2, struct)

%%%%% size
[~,features_num]=size(features);
[hidden_layer_n,input_size]=size(w1);
%%%%% init
w1_grad=zeros(hidden_layer_n,input_size);
b1_grad=zeros(1,hidden_layer_n);

for hidden_layer_i=1:hidden_layer_n
    for input_size_i=1:input_size+1 % +1 for b
        for features_num_i=1:features_num 
            %%%%% struct
            [z, z_sig, y, y_sig]=Unpack_Struct(struct(features_num_i));
            
            %%%%% cur parameters           
            cur_input=features(:,features_num_i);
            cur_label=labels(:,features_num_i);
            cur_z=z(hidden_layer_i);
            cur_w2=w2(hidden_layer_i);
            
            %%%%% mul
            mul1=Dif_Cost(y_sig, cur_label, features_num);
            mul2=Dif_Sigmoid(y);
            mul3=cur_w2;
            mul4=Dif_Sigmoid(cur_z);
            all_mul=mul1*mul2*mul3*mul4;
            
            %%%%% last mul
            if input_size_i~=input_size+1
                last_dif=cur_input(input_size_i);
                w1_grad(hidden_layer_i,input_size_i)=w1_grad(hidden_layer_i,input_size_i)+all_mul*last_dif;
            else % for b
                last_dif=1;
                b1_grad(hidden_layer_i)=b1_grad(hidden_layer_i)+all_mul*last_dif;
            end    
            
        end 
    end
    
end

b1_grad=b1_grad';


end