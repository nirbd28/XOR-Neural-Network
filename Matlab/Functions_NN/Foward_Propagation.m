function [output, struct]=Foward_Propagation(input, w1, b1, w2, b2)

z=w1*input+b1;
z_sig=Sigmoid(z)';
y=w2*z_sig+b2;
y_sig=Sigmoid(y);

output=y_sig;

%%%%% struct parameters
struct.z=z;
struct.z_sig=z_sig;
struct.y=y;
struct.y_sig=y_sig;

end