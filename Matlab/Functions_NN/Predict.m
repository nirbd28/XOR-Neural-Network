function recognized_output=Predict(input, w1, b1, w2, b2)

output=Foward_Propagation(input, w1, b1, w2, b2);

if output>=0.5
    recognized_output='1';
else
    recognized_output='0';
end

end