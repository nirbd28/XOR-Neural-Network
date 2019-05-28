######################################## Import from folder
folder_name='Functions_NN'
import sys
path_name=sys.path[0]
path_name_new=path_name+'\\'+folder_name
sys.path.append(path_name_new)
##### Import Files
import Main_Func,Dif
########################################

import matplotlib.pyplot as plt
import numpy as np
########################################

########## load data
features = np.loadtxt('features.txt', delimiter=',')
labels = np.loadtxt('labels.txt', delimiter=',')
### size
input_size,features_num=np.shape(features)
output_size=1

########## NN inputs
itter_num=2000
learning_rate=2
hidden_layer_n=3

##### init weights
w1 = np.random.normal(0, 1, (hidden_layer_n,input_size))
b1 = np.random.normal(0, 1, (hidden_layer_n,))
w2 = np.random.normal(0, 1, (output_size,hidden_layer_n))
b2 = np.random.normal(0, 1, (output_size,))
###
cost,_,_,_,_=Main_Func.Parameters_OPT(features, labels, w1, b1, w2, b2, learning_rate, itter_num)

##### plot
plt.plot(cost)
plt.xlabel('Itter')
plt.ylabel('Cost')
plt.title('Cost VS Itter')
plt.show()

##### check
Main_Func.Check_NN(features, labels, w1, b1, w2, b2)

### Hold CMD
input()