#!/bin/bash

python3 run.py \
--data=/nfs/nas-5.1/mxshi/2016_data/data_tester/sampler/General-Numeral-Attachment/EarningsCall_Dataset/list_data/full/data.json \
--train_data=/nfs/nas-5.1/mxshi/2016_data/data_tester/sampler/General-Numeral-Attachment/EarningsCall_Dataset/list_data/train/data.json \
--test_data=/nfs/nas-5.1/mxshi/2016_data/data_tester/sampler/General-Numeral-Attachment/EarningsCall_Dataset/list_data/test/data.json \
--mode=test \
--padding_length=512 \
--predict_day=$1 \
--padding_embed=384 \
--extend_token=250 \
--pretrained_model=/home/mxshi/General-Numeral-Attachment/reproduce/pre-trained/ckpt/checkpoints/epoch_ckpt_9 \
--batch_size=4 \
--seed=24 \
--model_dir=ckpt
