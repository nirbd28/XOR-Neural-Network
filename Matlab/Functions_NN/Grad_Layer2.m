function [w2_grad, b2_grad]=Grad_Layer2(features, labels, w1, b1, w2, b2, struct)

%%%%% size
[~,features_num]=size(features);
[hidden_layer_n,input_size]=size(w1);
output_size=length(b2);
%%%%% init
w2_grad=zeros(hidden_layer_n,output_size);
b2_grad=zeros(1,output_size);

for hidden_layer_i=1:hidden_layer_n+1 % +1 for b
        for features_num_i=1:features_num
            %%%%% struct
            [z, z_sig, y, y_sig]=Unpack_Struct(struct(features_num_i));
            
            %%%%% cur parameters           
            cur_label=labels(:,features_num_i);
            
            %%%%% mul
            mul1=Dif_Cost(y_sig, cur_label, features_num);
            mul2=Dif_Sigmoid(y);
            all_mul=mul1*mul2;
            
            %%%%% last mul
            if hidden_layer_i~=hidden_layer_n+1
                last_dif=z_sig(hidden_layer_i);
                w2_grad(hidden_layer_i)=w2_grad(hidden_layer_i)+all_mul*last_dif;
            else % for b
                last_dif=1;
                b2_grad(output_size)=b2_grad(output_size)+all_mul*last_dif;
            end    
            
        end 
end

w2_grad=w2_grad';

end