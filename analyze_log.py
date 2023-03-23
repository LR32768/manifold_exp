import numpy as np
import re

file_name = 'NTK_log_2.txt'
f = open(file_name)
line = f.readline()

#pattern = re.compile()
log_dict = {}

while line:
    num = line[12:-1]
    line = f.readline()
    data = line.split()
    data = np.array([float(x) for x in data])
    # print(num)
    # print(data)
    if not num in log_dict.keys():
        log_dict[num] = data
    else:
        log_dict[num] = np.vstack((log_dict[num], data))
    line = f.readline()
f.close()

np.set_printoptions(precision=5, suppress=True)
mean_tensor = None
std_tensor = None
for key in log_dict.keys():
    print(key)
    if mean_tensor is None:
        mean_tensor = log_dict[key].mean(axis=0)
    else:
        mean_tensor = np.vstack((mean_tensor, log_dict[key].mean(axis=0)))

    if std_tensor is None:
        std_tensor = log_dict[key].std(axis=0)
    else:
        std_tensor = np.vstack((std_tensor, log_dict[key].std(axis=0)))

print(mean_tensor.transpose())
print(std_tensor.transpose())