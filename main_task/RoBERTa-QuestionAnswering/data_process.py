import json
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class paragraphDataset(Dataset):
	
	def __init__(self, inputs, numeral_position, start_position, end_position):
		self.inputs = torch.LongTensor(inputs)
		self.numeral_position = numeral_position
		self.start_position = torch.LongTensor(start_position)
		self.end_position = torch.LongTensor(end_position)
	
	def __len__(self):
		return len(self.inputs)
	
	def __getitem__(self, idx):
		return self.inputs[idx], self.numeral_position[idx], self.start_position[idx], self.end_position[idx]

def collate_fn(batch):
	batch_inputs, batch_numeral_position, batch_start_position, batch_end_position = zip(*batch)
	return torch.stack(batch_inputs), batch_numeral_position, torch.stack(batch_start_position), torch.stack(batch_end_position)

def get_train_dataloader(args, tokenizer):
	with open(args['train_data'], 'r') as fp:
		data = json.load(fp)
	inputs, numeral_position, start_position, end_position = [], [], [], []

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

		name_entity = d['name_entity']
		input_sentence = d['paragraph'][offset_start: d['offset_start']] + " " + tokenizer.mask_token + " " + d['paragraph'][d['offset_end']: offset_end]
		original_start_position = -1
		target_position = d['offset_start'] - offset_start

		for idx in range(len(input_sentence) - len(name_entity)):
			if input_sentence[idx: idx + len(name_entity)] == name_entity and (original_start_position ==-1 or abs(target_position - idx) < abs(target_position - original_start_position)):
				original_start_position = idx
		
		original_end_position = original_start_position + len(name_entity)
		

		input_tokens = tokenizer(input_sentence, return_tensors='np', padding='max_length', max_length=args['padding_length'], return_attention_mask=False, return_offsets_mapping=True)
		numeral_p = 0
		for idx, token_id in enumerate(input_tokens.input_ids[0]):
			if token_id == tokenizer.mask_token_id:
				numeral_p = idx
				break

		L, R = -1, -1
		for idx, offset in enumerate(input_tokens.offset_mapping[0]):
			if offset[0] <= original_start_position and original_start_position < offset[1]:
				L = idx
			if offset[0] < original_end_position and original_end_position <= offset[1]:
				R = idx

		if name_entity == 'None':
			L = R = 0 # set to [CLS] token
		if L == -1 or R == -1:
			continue # some issue in data. print input sentence and name_entity to get more info.

		inputs.append(input_tokens.input_ids[0])
		numeral_position.append(numeral_p)
		start_position.append(L)
		end_position.append(R)
	
	train_dataset = paragraphDataset(inputs, numeral_position, start_position, end_position)
	train_dataloader = DataLoader(
						train_dataset, 
						args['batch_size'], 
						shuffle=True, 
						collate_fn=collate_fn, 
						num_workers=4, 
					)
	return train_dataloader

def get_eval_dataloader(args, tokenizer):
	with open(args['eval_data'], 'r') as fp:
		data = json.load(fp)
	inputs, numeral_position, start_position, end_position = [], [], [], []

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

		name_entity = d['name_entity']
		input_sentence = d['paragraph'][offset_start: d['offset_start']] + " " + tokenizer.mask_token + " " + d['paragraph'][d['offset_end']: offset_end]
		original_start_position = -1
		target_position = d['offset_start'] - offset_start

		for idx in range(len(input_sentence) - len(name_entity)):
			if input_sentence[idx: idx + len(name_entity)] == name_entity and (original_start_position ==-1 or abs(target_position - idx) < abs(target_position - original_start_position)):
				original_start_position = idx
		
		original_end_position = original_start_position + len(name_entity)
		

		input_tokens = tokenizer(input_sentence, return_tensors='np', padding='max_length', max_length=args['padding_length'], return_attention_mask=False, return_offsets_mapping=True)
		numeral_p = 0
		for idx, token_id in enumerate(input_tokens.input_ids[0]):
			if token_id == tokenizer.mask_token_id:
				numeral_p = idx
				break

		L, R = -1, -1
		for idx, offset in enumerate(input_tokens.offset_mapping[0]):
			if offset[0] <= original_start_position and original_start_position < offset[1]:
				L = idx
			if offset[0] < original_end_position and original_end_position <= offset[1]:
				R = idx

		if name_entity == 'None':
			L = R = 0 # set to [CLS] token
		if L == -1 or R == -1:
			continue # some issue in data. print input sentence and name_entity to get more info.

		inputs.append(input_tokens.input_ids[0])
		numeral_position.append(numeral_p)
		start_position.append(L)
		end_position.append(R)
	
	eval_dataset = paragraphDataset(inputs, numeral_position, start_position, end_position)
	eval_dataloader = DataLoader(
						eval_dataset, 
						args['batch_size'], 
						shuffle=False, 
						collate_fn=collate_fn, 
						num_workers=4, 
					)
	return eval_dataloader

def get_test_dataloader(args, tokenizer):
	with open(args['test_data'], 'r') as fp:
		data = json.load(fp)
	inputs, numeral_position, start_position, end_position = [], [], [], []

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

		name_entity = d['name_entity']
		input_sentence = d['paragraph'][offset_start: d['offset_start']] + " " + tokenizer.mask_token + " " + d['paragraph'][d['offset_end']: offset_end]
		original_start_position = -1
		target_position = d['offset_start'] - offset_start

		for idx in range(len(input_sentence) - len(name_entity)):
			if input_sentence[idx: idx + len(name_entity)] == name_entity and (original_start_position ==-1 or abs(target_position - idx) < abs(target_position - original_start_position)):
				original_start_position = idx
		
		original_end_position = original_start_position + len(name_entity)
		

		input_tokens = tokenizer(input_sentence, return_tensors='np', padding='max_length', max_length=args['padding_length'], return_attention_mask=False, return_offsets_mapping=True)
		numeral_p = 0
		for idx, token_id in enumerate(input_tokens.input_ids[0]):
			if token_id == tokenizer.mask_token_id:
				numeral_p = idx
				break

		L, R = -1, -1
		for idx, offset in enumerate(input_tokens.offset_mapping[0]):
			if offset[0] <= original_start_position and original_start_position < offset[1]:
				L = idx
			if offset[0] < original_end_position and original_end_position <= offset[1]:
				R = idx

		if name_entity == 'None':
			L = R = 0 # set to [CLS] token
		if L == -1 or R == -1:
			continue # some issue in data. print input sentence and name_entity to get more info.

		inputs.append(input_tokens.input_ids[0])
		numeral_position.append(numeral_p)
		start_position.append(L)
		end_position.append(R)
	
	test_dataset = paragraphDataset(inputs, numeral_position, start_position, end_position)
	test_dataloader = DataLoader(
						test_dataset, 
						args['batch_size'], 
						shuffle=False, 
						collate_fn=collate_fn, 
						num_workers=4, 
					)
	return test_dataloader
