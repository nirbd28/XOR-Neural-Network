function succes_rate=Check_NN(features, labels, w1, b1, w2, b2)

[input_size,features_num]=size(features);

succes_count=0;
for i=1:features_num
    %%% cur input/label
    input=features(:,i);
    label=labels(i);
    %%% calc output
    recognized_output=Predict(input, w1, b1, w2, b2);
    %%%%% strings
    label_str=num2str(label);
    recognized_output_str=num2str(recognized_output);
    %%% strcat input
    input_str=[];
    for i=input_size:-1:1
        input_str=strcat(input_str,num2str(input(i)));
    end
    %%%%% compare
    if str2num(recognized_output)==label
       str='good!!!!!' ;
       succes_count=succes_count+1;
    else
        str='bad :(';
    end
    %%% display
    disp(strcat('Input=',input_str,',','Label=', label_str,',','Rec Output=',recognized_output_str,',', str));
end
%%% statistics

succes_rate=(succes_count/features_num)*100;
disp(strcat('succes rate=',num2str(succes_rate),'%'))

end