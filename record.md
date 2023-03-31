* Covid dataset (node regression)

| Sparsity | 5% | 10% | 20% |
| ---- | ---- | ---- | ---- |
| MAE w/ hetero&temporal embed | 7.330 | 3.686 | 3.270 |
| MAE w/o hetero&temporal embed | 7.541 | 5.198 | 3.394 |

* MAG dataset (link prediction)

| Sparsity | 2% | 5% | 8% |
| ---- | ---- | ---- | ---- |
| AUC w/ hetero&temporal embed | 0.960 | 0.976 | 0.981 |
| AUC w/o hetero&temporal embed | 0.947 | 0.967 | 0.975 |
| AP w/ hetero&temporal embed | 0.930 | 0.961 | 0.971 |
| AP w/o hetero&temporal embed | 0.901 | 0.943 | 0.963 |
  
  
  
## covid w/o hetero&temporal embed
0.05 0.1 0.2  
35.9203878  34.97489282 31.1581859  
36.56946382 35.70362779 30.93287954  
36.09224949 34.55087751 30.50287878

## covid w/ hetero&temporal embed
19.31965011 12.98986259  6.7636781
18.43877813 12.53882574  6.76109315




## covid w/ hetero&temporal embed 
node regression
see folder covid_test


## covid w/o  
lr 0.005  
see folder covid_test1

## mag w/  
lr 0.001  
see folder mag_test

## mag w/o  
lr 0.001  
see folder mag_test1

