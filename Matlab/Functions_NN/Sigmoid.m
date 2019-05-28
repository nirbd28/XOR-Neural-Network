function y=Sigmoid(x)

for i=1:length(x)
    y(i)=1/(1+exp(- x(i)));  
end

end