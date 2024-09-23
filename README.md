[Progressive Skeleton Learning for Effective Local-to-Global Causal Structure Learning](https://ieeexplore.ieee.org/abstract/document/10681652) <br>

# Usage
"PCSL.m" is main function. <br>
Note that the current code has only been debugged on Matlab (2018a) with a 64-bit Windows system and supports only discrete datasets.<br>
----------------------------------------------
function [DAG, time] = PCSL(Data_0, Alpha, rand_sample_numb) <br>
* INPUT: <br>
```Matlab
Data_0 is the data matrix, and rows represent the number of samples and columns represent the number of nodes. If Data_0 is a discrete dataset, the value in Data_0 should start from 1.
Alpha is the significance level, e.g., 0.01 or 0.05.
rand_sample_numb is the number of sub-datasets generated, e.g., 15.
```
* OUTPUT: <br>
```Matlab
DAG is a directed acyclic graph learned on a given dataset。
time is the runtime of the algorithm.
```

# Example for discrete dataset
```Matlab
clear;
clc;
addpath(genpath('common_func/'));
alpha=0.01;
data=load('./dataset/Alarm_1000s.txt');
data=data+1;
[DAG, time] = PCSL(data, alpha, 15);
```

# Reference
* Guo, Xianjie, et al. "Progressive Skeleton Learning for Effective Local-to-Global Causal Structure Learning." *IEEE Transactions on Knowledge and Data Engineering (TKDE)* (2024).
