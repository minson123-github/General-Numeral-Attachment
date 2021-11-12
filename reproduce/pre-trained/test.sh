#!/bin/bash

python3 run.py \
--data=/nfs/nas-5.1/mxshi/2016_data/data_tester/sampler/General-Numeral-Attachment/data/data.json \
--train_data=/nfs/nas-5.1/mxshi/2016_data/data_tester/sampler/General-Numeral-Attachment/data/train.json \
--eval_data=/nfs/nas-5.1/mxshi/2016_data/data_tester/sampler/General-Numeral-Attachment/data/eval.json \
--test_data=/nfs/nas-5.1/mxshi/2016_data/data_tester/sampler/General-Numeral-Attachment/data/test.json \
--mode=test \
--batch_size=4 \
--extend_token=250 \
--padding_length=512 \
--testing_model=ckpt/checkpoints/epoch_ckpt_9 \
--seed=24 \
--model_dir=ckpt 
