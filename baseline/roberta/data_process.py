import json
import torch
from tqdm import tqdm
from transformers import T5TokenizerFast
from torch.utils.data import DataLoader, Dataset

class paragraphDataset(Dataset):
	
	def __init__(self, inputs, labels = None):
		self.inputs = torch.LongTensor(inputs)
		self.labels = torch.LongTensor(labels)
	
	def __len__(self):
		return len(self.inputs)
	
	def __getitem__(self, idx):
		return self.inputs[idx], self.labels[idx]

def collate_fn(batch):
	batch_inputs, batch_labels = zip(*batch)
	return torch.stack(batch_inputs), torch.stack(batch_labels)

def get_all_name_entity(args):
	with open(args['data'], 'r') as fp:
		data = json.load(fp)
	name_entity_set = set()
	for d in data:
		name_entity_set.add(d['name_entity'])
	return [name_entity for name_entity in name_entity_set]

def get_train_dataloader(args, tokenizer):
	name_entity = get_all_name_entity(args)
	entity2id = {entity: i for i, entity in enumerate(name_entity)}
	with open(args['train_data'], 'r') as fp:
		data = json.load(fp)
	inputs, labels = [], []
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

		input_sentence = d['target_numeral'] + tokenizer.sep_token + d['paragraph'][offset_start: offset_end]
		entity = d['name_entity']

		input_tokens = tokenizer(input_sentence, return_tensors='np', padding='max_length', return_attention_mask=False)
		label = entity2id[entity]
		inputs.append(input_tokens.input_ids[0])
		labels.append(label)
	
	train_dataset = paragraphDataset(inputs, labels)
	train_dataloader = DataLoader(
						train_dataset, 
						args['batch_size'], 
						shuffle=True, 
						collate_fn=collate_fn, 
						num_workers=args['n_gpu'] * 4
					)
	return train_dataloader

def get_eval_dataloader(args, tokenizer):
	name_entity = get_all_name_entity(args)
	entity2id = {entity: i for i, entity in enumerate(name_entity)}
	with open(args['eval_data'], 'r') as fp:
		data = json.load(fp)
	inputs, labels = [], []
	for d in tqdm(data, position = 0, desc='tokenize'):
		paragraph_tokens = tokenizer(d['paragraph'], return_tensors='np', return_offsets_mapping=True, return_attention_mask=False)
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

		input_sentence = d['target_numeral'] + tokenizer.sep_token + d['paragraph'][offset_start: offset_end]
		entity = d['name_entity']

		input_tokens = tokenizer(input_sentence, return_tensors='np', padding='max_length', return_attention_mask=False)
		label = entity2id[entity]
		inputs.append(input_tokens.input_ids[0])
		labels.append(label)
	
	eval_dataset = paragraphDataset(inputs, labels)
	eval_dataloader = DataLoader(
						eval_dataset, 
						args['batch_size'], 
						shuffle=False, 
						collate_fn=collate_fn, 
						num_workers=args['n_gpu'] * 4
					)
	return eval_dataloader

def get_test_dataloader(args, tokenizer):
	name_entity = get_all_name_entity(args)
	entity2id = {entity: i for i, entity in enumerate(name_entity)}
	with open(args['test_data'], 'r') as fp:
		data = json.load(fp)
	inputs, labels = [], []
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

		input_sentence = d['target_numeral'] + tokenizer.sep_token + d['paragraph'][offset_start: offset_end]
		entity = d['name_entity']

		input_tokens = tokenizer(input_sentence, return_tensors='np', padding='max_length', return_attention_mask=False)
		label = entity2id[entity]
		inputs.append(input_tokens.input_ids[0])
		labels.append(label)
	
	test_dataset = paragraphDataset(inputs, labels)
	test_dataloader = DataLoader(
						test_dataset, 
						args['batch_size'], 
						shuffle=False, 
						collate_fn=collate_fn, 
						num_workers=4
					)
	return test_dataloader
