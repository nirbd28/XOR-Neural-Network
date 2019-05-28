%%
clc; clear all; close all;
%%
clc; clear all; close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input_size=3;

for d=0:(2^input_size-1)
   features(:,d+1)=de2bi(d,input_size)';
   xor_sum=0;
   for cur_bit_i=1:input_size
       cur_bit=features(cur_bit_i,d+1);
       xor_sum=bitxor(xor_sum,cur_bit);
   end
   labels(d+1)=xor_sum; 
end

save('features.mat','features')
save('labels.mat','labels')

