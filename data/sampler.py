import os
import json
import random
import argparse

def get_config():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, help='The directory contains data.')
	parser.add_argument('--seed', type=int, default=24, help='random seed.')
	parser.add_argument('--train_ratio', type=int, help='Ratio of training data.')
	parser.add_argument('--eval_ratio', type=int, help='Ratio of evaluate data.')
	parser.add_argument('--test_ratio', type=int, help='Ratio of testing data.')
	args = parser.parse_args()
	return vars(args)

def sample_data(args):
	total_ratio = args['train_ratio'] + args['eval_ratio'] + args['test_ratio']
	with open(os.path.join(args['data_dir'], 'data.json'), 'r') as fp:
		data = json.load(fp)
	train_data, eval_data, test_data = [], [], []
	random.seed(args['seed'])
	random.shuffle(data)
	for i, d in enumerate(data):
		if i % total_ratio < args['train_ratio']:
			train_data.append(d)
		elif i % total_ratio < args['train_ratio'] + args['eval_ratio']:
			eval_data.append(d)
		else:
			test_data.append(d)
	
	with open(os.path.join(args['data_dir'], 'train.json'), 'w') as fp:
		json.dump(train_data, fp)
	with open(os.path.join(args['data_dir'], 'eval.json'), 'w') as fp:
		json.dump(eval_data, fp)
	with open(os.path.join(args['data_dir'], 'test.json'), 'w') as fp:
		json.dump(test_data, fp)

if __name__ == '__main__':
	args = get_config()
	sample_data(args)
