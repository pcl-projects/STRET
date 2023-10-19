import numpy as np
import pandas as pd

import time

all_computation_time ={}

all_batch_size = {}

opt_batch = {}
num_workers =3

min_comp = 9999
min_index = 0




def close_to_any(a, floats, **kwargs):
    return np.where(np.isclose(a, floats, **kwargs))

start = time.time()

for i in range(0, num_workers):
    file_bsearch = 'resnet_my_method/worker_'+str(i)+'.csv'
    df_bsearch = pd.read_csv(file_bsearch, header=None)
#     # print(df_itn)
   
    index_worker_all = pd.DataFrame(df_bsearch.values[:,0])
#     # print(itn_time_all)

    batch_all = pd.DataFrame(df_bsearch.values[:,1])
    computation_time_all = pd.DataFrame(df_bsearch.values[:,2])

#     communication_time_all = pd.DataFrame(df_cc_time.values[:,2])
#     # print(itn_time_all)

    computation_time_all = computation_time_all.to_numpy()
    batch_all = batch_all.to_numpy()
    all_computation_time[i] = computation_time_all
    all_batch_size[i] = batch_all
    print(len(computation_time_all))

    if min_comp > computation_time_all[len(computation_time_all)-1]:
        min_comp = computation_time_all[len(computation_time_all)-1][0]
        min_index = i
        print(min_comp)

# for checking if the float value is any closer to other values
# print("Hello world")
# import numpy as np 

# def close_to_any(a, floats, **kwargs):
#   return np.any(np.isclose(a, floats, **kwargs))
  
# print(close_to_any(0.3, [0.6, 0.4, 0.32], atol=0.01))

# for checking and geting the index of the position where the float value is closer
# print("Hello world")
# import numpy as np 

# def close_to_any(a, floats, **kwargs):
#   return np.where(np.isclose(a, floats, **kwargs))
  
# i, = close_to_any(0.3, [0.6, 0.31, 0.32], atol=0.01)
# print(i[0])




for j in range(0, num_workers):
    if j != min_index:
        get_array = np.array(all_computation_time[j])
        get_index = close_to_any(min_comp, get_array, atol=1.0)

        index_correct = get_index[0][len(get_index)-1]
        print(index_correct)
        # get_index = get_index[0]
        opt_batch[j] = all_batch_size[j][index_correct][0]

end = time.time()
print("time "+str(end - start))

print(opt_batch)












