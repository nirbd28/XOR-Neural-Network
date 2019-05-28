function cost=Calc_Cost(labels, output)

features_num=length(labels);

cost=0;
for features_num_i=1:features_num
   cur_label=labels(features_num_i);
   cur_output=output(features_num_i);
   cost=cost+MSE(cur_output, cur_label);
end
cost=cost/features_num;

end