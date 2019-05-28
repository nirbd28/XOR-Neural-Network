######################################## Import from folder
folder_name='Functions_NN'
import sys
path_name=sys.path[0]
path_name_new=path_name+'\\'+folder_name
sys.path.append(path_name_new)
##### Import Files
import Main_Func,Dif
########################################

import numpy as np
####################

input_size=3
### init
features=np.zeros((input_size,np.power(2,input_size)))
labels=np.zeros(np.power(2,input_size))
###
for d in range(0,np.power(2,input_size)):
    cur_input=Main_Func.int_to_bin(d,input_size)
    sum_xor='0'
    for i in range(0,input_size):
        features[i,d]=cur_input[i]
        sum_xor=Main_Func.logical_xor(sum_xor,cur_input[i])
    labels[d]=sum_xor
### save txt files
np.savetxt('features.txt', features,fmt='%0.0f', delimiter=',')
np.savetxt('labels.txt', labels,fmt='%0.0f', delimiter=',')
