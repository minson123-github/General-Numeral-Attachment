import json
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class paragraphDataset(Dataset):
	
	def __init__(self, inputs, numeral_position, sign_id):
		self.inputs = torch.LongTensor(inputs)
		self.numeral_position = numeral_position
		self.sign_id = torch.LongTensor(sign_id)
	
	def __len__(self):
		return len(self.inputs)
	
	def __getitem__(self, idx):
		return self.inputs[idx], self.numeral_position[idx], self.sign_id[idx]

def collate_fn(batch):
	batch_inputs, batch_numeral_position, batch_sign_id = zip(*batch)
	return torch.stack(batch_inputs), batch_numeral_position, torch.stack(batch_sign_id)

def get_all_sign(args):
	with open(args['data'], 'r') as fp:
		data = json.load(fp)
	sign_set = set()
	for d in data:
		sign_set.add(d['sign'])
	return sorted([sign for sign in sign_set])

def get_train_dataloader(args, tokenizer):
	sign = get_all_sign(args)
	sign2id = {sign: i for i, sign in enumerate(sign)}
	with open(args['train_data'], 'r') as fp:
		data = json.load(fp)
	inputs, numeral_position, sign_id = [], [], []

	for d in tqdm(data, position = 0, desc='tokenize'):
		paragraph_tokens = tokenizer(d['paragraph'], return_tensors='np', return_offsets_mapping=True, return_attention_mask=False, verbose=False)
		L, R = -1, -1
		for idx, offset in enumerate(paragraph_tokens.offset_mapping[0]):
			if offset[0] <= d['offset_start'] and d['offset_start'] < offset[1]:
				L = idx
			if offset[0] < d['offset_end'] and d['offset_end'] <= offset[1]:
				R = idx
		offset_start, offset_end = -1, -1
		for idx, offset in enumerate(paragraph_tokens.offset_mapping[0]):
			if idx >= L - args['extend_token'] and (offset[0] != 0 or offset[1] != 0) and offset_start == -1:
				offset_start = offset[0]
			if idx <= R + args['extend_token'] and (offset[0] != 0 or offset[1] != 0):
				offset_end = offset[1]

		s = sign2id[d['sign']]
		input_sentence = d['paragraph'][offset_start: d['offset_start']] + " " + tokenizer.mask_token + " " + d['paragraph'][d['offset_end']: offset_end]

		input_tokens = tokenizer(input_sentence, return_tensors='np', padding='max_length', max_length=args['padding_length'], return_attention_mask=False)
		numeral_p = 0
		for idx, token_id in enumerate(input_tokens.input_ids[0]):
			if token_id == tokenizer.mask_token_id:
				numeral_p = idx
				break

		inputs.append(input_tokens.input_ids[0])
		numeral_position.append(numeral_p)
		sign_id.append(s)
	
	train_dataset = paragraphDataset(inputs, numeral_position, sign_id)
	train_dataloader = DataLoader(
						train_dataset, 
						args['batch_size'], 
						shuffle=True, 
						collate_fn=collate_fn, 
						num_workers=4, 
					)
	return train_dataloader

def get_eval_dataloader(args, tokenizer):
	sign = get_all_sign(args)
	sign2id = {sign: i for i, sign in enumerate(sign)}
	with open(args['eval_data'], 'r') as fp:
		data = json.load(fp)
	inputs, numeral_position, sign_id = [], [], []

	for d in tqdm(data, position = 0, desc='tokenize'):
		paragraph_tokens = tokenizer(d['paragraph'], return_tensors='np', return_offsets_mapping=True, return_attention_mask=False, verbose=False)
		L, R = -1, -1
		for idx, offset in enumerate(paragraph_tokens.offset_mapping[0]):
			if offset[0] <= d['offset_start'] and d['offset_start'] < offset[1]:
				L = idx
			if offset[0] < d['offset_end'] and d['offset_end'] <= offset[1]:
				R = idx
		offset_start, offset_end = -1, -1
		for idx, offset in enumerate(paragraph_tokens.offset_mapping[0]):
			if idx >= L - args['extend_token'] and (offset[0] != 0 or offset[1] != 0) and offset_start == -1:
				offset_start = offset[0]
			if idx <= R + args['extend_token'] and (offset[0] != 0 or offset[1] != 0):
				offset_end = offset[1]

		s = sign2id[d['sign']]
		input_sentence = d['paragraph'][offset_start: d['offset_start']] + " " + tokenizer.mask_token + " " + d['paragraph'][d['offset_end']: offset_end]

		input_tokens = tokenizer(input_sentence, return_tensors='np', padding='max_length', max_length=args['padding_length'], return_attention_mask=False)
		numeral_p = 0
		for idx, token_id in enumerate(input_tokens.input_ids[0]):
			if token_id == tokenizer.mask_token_id:
				numeral_p = idx
				break

		inputs.append(input_tokens.input_ids[0])
		numeral_position.append(numeral_p)
		sign_id.append(s)
	
	eval_dataset = paragraphDataset(inputs, numeral_position, sign_id)
	eval_dataloader = DataLoader(
						eval_dataset, 
						args['batch_size'], 
						shuffle=False, 
						collate_fn=collate_fn, 
						num_workers=4, 
					)
	return eval_dataloader

def get_test_dataloader(args, tokenizer):
	sign = get_all_sign(args)
	sign2id = {sign: i for i, sign in enumerate(sign)}
	with open(args['test_data'], 'r') as fp:
		data = json.load(fp)
	inputs, numeral_position, sign_id = [], [], []

	for d in tqdm(data, position = 0, desc='tokenize'):
		paragraph_tokens = tokenizer(d['paragraph'], return_tensors='np', return_offsets_mapping=True, return_attention_mask=False, verbose=False)
		L, R = -1, -1
		for idx, offset in enumerate(paragraph_tokens.offset_mapping[0]):
			if offset[0] <= d['offset_start'] and d['offset_start'] < offset[1]:
				L = idx
			if offset[0] < d['offset_end'] and d['offset_end'] <= offset[1]:
				R = idx
		offset_start, offset_end = -1, -1
		for idx, offset in enumerate(paragraph_tokens.offset_mapping[0]):
			if idx >= L - args['extend_token'] and (offset[0] != 0 or offset[1] != 0) and offset_start == -1:
				offset_start = offset[0]
			if idx <= R + args['extend_token'] and (offset[0] != 0 or offset[1] != 0):
				offset_end = offset[1]

		s = sign2id[d['sign']]
		input_sentence = d['paragraph'][offset_start: d['offset_start']] + " " + tokenizer.mask_token + " " + d['paragraph'][d['offset_end']: offset_end]

		input_tokens = tokenizer(input_sentence, return_tensors='np', padding='max_length', max_length=args['padding_length'], return_attention_mask=False)
		numeral_p = 0
		for idx, token_id in enumerate(input_tokens.input_ids[0]):
			if token_id == tokenizer.mask_token_id:
				numeral_p = idx
				break

		inputs.append(input_tokens.input_ids[0])
		numeral_position.append(numeral_p)
		sign_id.append(s)
	
	test_dataset = paragraphDataset(inputs, numeral_position, sign_id)
	test_dataloader = DataLoader(
						test_dataset, 
						args['batch_size'], 
						shuffle=False, 
						collate_fn=collate_fn, 
						num_workers=4, 
					)
	return test_dataloader
