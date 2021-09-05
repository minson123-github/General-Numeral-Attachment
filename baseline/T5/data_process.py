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

def get_train_dataloader(args, tokenizer):
	with open(args['train_data'], 'r') as fp:
		data = json.load(fp)
	inputs, labels = [], []
	for d in tqdm(data, position = 0, desc='tokenize'):
		input_sentence = d['paragraph'] + tokenizer.sep_token + d['target_numeral']
		name_entity = d['name_entity']
		input_tokens = tokenizer(input_sentence, return_tensors='np', padding='max_length', max_length=1024, return_attention_mask=False)
		label_tokens = tokenizer(name_entity, return_tensors='np', padding='max_length', max_length=1024, return_attention_mask=False)
		inputs.append(input_tokens.input_ids[0])
		labels.append(label_tokens.input_ids[0])
	
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
	with open(args['eval_data'], 'r') as fp:
		data = json.load(fp)
	inputs, labels = [], []
	for d in tqdm(data, position = 0, desc='tokenize'):
		input_sentence = d['paragraph'] + tokenizer.sep_token + d['target_numeral']
		name_entity = d['name_entity']
		input_tokens = tokenizer(input_sentence, return_tensors='np', padding='max_length', max_length=1024, return_attention_mask=False)
		label_tokens = tokenizer(name_entity, return_tensors='np', padding='max_length', max_length=1024, return_attention_mask=False)
		inputs.append(input_tokens.input_ids[0])
		labels.append(label_tokens.input_ids[0])
	
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
	with open(args['test_data'], 'r') as fp:
		data = json.load(fp)
	inputs, labels = [], []
	for d in tqdm(data, position = 0, desc='tokenize'):
		input_sentence = d['paragraph'] + tokenizer.sep_token + d['target_numeral']
		name_entity = d['name_entity']
		input_tokens = tokenizer(input_sentence, return_tensors='np', padding='max_length', max_length=1024, return_attention_mask=False)
		label_tokens = tokenizer(name_entity, return_tensors='np', padding='max_length', max_length=1024, return_attention_mask=False)
		inputs.append(input_tokens.input_ids[0])
		labels.append(label_tokens.input_ids[0])
	
	test_dataset = paragraphDataset(inputs, labels)
	test_dataloader = DataLoader(
						test_dataset, 
						args['batch_size'], 
						shuffle=False, 
						collate_fn=collate_fn, 
						num_workers=args['n_gpu'] * 4
					)
	return test_dataloader
