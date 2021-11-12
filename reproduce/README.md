The following is our reproduce environment, if you want get exactly value as we report in the paper, make sure you have the same environment.

pytorch version: 1.9.0+cu111
GPU: RTX 3090
cuda version: 11.2
python version: 3.8.11

Due to the non-determinism issue for some RNN related model, make sure you set the following environment variable to get the correct reproduce results.

On CUDA 10.1, set environment variable CUDA_LAUNCH_BLOCKING=1. This may affect performance.

On CUDA 10.2 or later, set environment variable (note the leading colon symbol) CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:2.

# How to reproduce results
For each baseline model, we write train.sh/test.sh for it. If you want to reproduce the model focus on predict one label(e.g. single, roberta). Please run: 
```
bash train.sh $predict-day
```
and
```
bash test.sh $predict-day
```
In here, $predict-day represent the day of information you want to predict.

Otherwise, you just run:
```
bash train.sh
```
and
```
bash test.sh
```

If there are any reproduce issue, please contact us.
