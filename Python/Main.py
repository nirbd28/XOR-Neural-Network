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

########## Run
run_times=100
cost=np.zeros((run_times,itter_num))
for i in range(0,run_times):
    ##### init weights
    w1 = np.random.normal(0, 1, (hidden_layer_n,input_size))
    b1 = np.random.normal(0, 1, (hidden_layer_n,))
    w2 = np.random.normal(0, 1, (output_size,hidden_layer_n))
    b2 = np.random.normal(0, 1, (output_size,))
    ###
    cost[i,:],_,_,_,_=Main_Func.Parameters_OPT(features, labels, w1, b1, w2, b2, learning_rate, itter_num)

##### calc mean cost
mean_cost=np.zeros(itter_num)
for i in range(0,itter_num):
    mean_cost[i]=np.mean(cost[:,i])

##### calc 95 percent
start_value=mean_cost[1]
finish_value=mean_cost[itter_num-1]
diff_1_perecnt=abs(start_value-finish_value)/100
diff_95_perecnt=finish_value+diff_1_perecnt*5

##### find 95 percent

for i in range(0,itter_num):
    if (mean_cost[i] <= diff_95_perecnt):
        diff_95_perecnt=mean_cost[i]
        itter_95=i
        break

##### plot
plt.plot( mean_cost, label='Mean Cost')
plt.xlabel('Itter')
plt.ylabel('Cost')
plt.title('Cost VS Itter')
plt.scatter(itter_95,mean_cost[itter_95],marker='*',color='red', label='%95')
plt.legend()
plt.show()

### Hold CMD
input()