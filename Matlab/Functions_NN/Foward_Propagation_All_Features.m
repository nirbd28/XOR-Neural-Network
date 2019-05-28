function [output, struct]=Foward_Propagation_All_Features(features, w1, b1, w2, b2)

[~,features_num]=size(features);

for features_num_i=1:features_num
    input=features(:,features_num_i);
    [output(features_num_i), struct(features_num_i)]=Foward_Propagation(input, w1, b1, w2, b2);
end

end