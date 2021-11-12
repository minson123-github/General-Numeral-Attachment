#!/bin/bash

python3 run.py \
--data=/nfs/nas-5.1/mxshi/2016_data/data_tester/sampler/General-Numeral-Attachment/data/data.json \
--train_data=/nfs/nas-5.1/mxshi/2016_data/data_tester/sampler/General-Numeral-Attachment/data/train.json \
--eval_data=/nfs/nas-5.1/mxshi/2016_data/data_tester/sampler/General-Numeral-Attachment/data/eval.json \
--test_data=/nfs/nas-5.1/mxshi/2016_data/data_tester/sampler/General-Numeral-Attachment/data/test.json \
--mode=train \
--batch_size=4 \
--extend_token=250 \
--padding_length=512 \
--lr=1e-5 \
--n_epoch=10 \
--seed=24 \
--saving_epochs=1 \
--model_dir=ckpt 
