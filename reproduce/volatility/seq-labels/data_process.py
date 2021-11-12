import os
import json
import torch
import random
import pickle
import string
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class paragraphDataset(Dataset):
    
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def collate_fn(batch):
    batch_embeds, batch_labels = zip(*batch)
    return torch.stack(batch_embeds), torch.stack(batch_labels)

def get_numeral_idx(sentences):
    numeral_idx = []
    numeral = '0123456789'
    start_idx = -1;
    end_idx = -1;
    for idx, c in enumerate(sentences):
        if c in numeral:
            if start_idx == -1:
                start_idx = idx
            end_idx = idx
        elif c not in string.punctuation:
            if start_idx != -1:
                numeral_idx.append((start_idx, end_idx + 1))
            start_idx = end_idx = -1;

    return numeral_idx

def get_train_dataloader(args, tokenizer, embed_model):
    with open(args['train_data'], 'r') as fp:
        data = json.load(fp)

    if os.path.exists('train_embed.pickle'):
        with open('train_embed.pickle', 'rb') as fp:
            all_embeds = pickle.load(fp)
    else:
        all_embeds = {}

    inputs, labels = [], []

    embed_model.to('cuda')
    embed_model.eval()
    empty_tokens = tokenizer('', return_tensors='pt', padding='max_length', max_length=args['padding_length'], return_attention_mask=False, return_offsets_mapping=False)
    empty_embed = embed_model(empty_tokens.input_ids.cuda()).last_hidden_state[:, 0, :].cpu()
    
    with torch.no_grad():
        for k, v in tqdm(data.items(), position = 0, desc='tokenize'):
            sentences = ''
            for sent in v['text']:
                sentences += sent
            try:
                embeds = all_embeds[k]
                while len(embeds) < args['padding_embed']:
                    embeds = torch.cat((embeds, empty_embed), dim=0)
            except KeyError:
                embeds = []
                numeral_idx = get_numeral_idx(sentences)
                all_tokens = tokenizer(sentences, return_tensors='np', return_offsets_mapping=True, return_attention_mask=False, verbose=False)
                for (numeral_l, numeral_r) in numeral_idx:
                    L, R = -1, -1
                    for idx, offset in enumerate(all_tokens.offset_mapping[0]):
                        if offset[0] <= numeral_l and numeral_l < offset[1]:
                            L = idx
                        if offset[0] < numeral_r and numeral_r <= offset[1]:
                            R = idx
                    offset_start, offset_end = -1, -1
                    for idx, offset in enumerate(all_tokens.offset_mapping[0]):
                        if idx >= L - args['extend_token'] and (offset[0] != 0 or offset[1] != 0) and offset_start == -1:
                            offset_start = offset[0]
                        if idx <= R + args['extend_token'] and (offset[0] != 0 or offset[1] != 0):
                            offset_end = offset[1]
                    input_sentence = sentences[offset_start: numeral_l] + ' ' + tokenizer.mask_token + ' ' + sentences[numeral_r: offset_end]
                    input_tokens = tokenizer(input_sentence, return_tensors='pt', padding='max_length', max_length=args['padding_length'], return_attention_mask=False, return_offsets_mapping=False)
                    numeral_p = 0 
                    for idx, token_id in enumerate(input_tokens.input_ids[0]):
                        if token_id == tokenizer.mask_token_id:
                            numeral_p = idx
                            break
                    assert len(input_tokens.input_ids) == 1
                    embed = embed_model(input_tokens.input_ids.cuda()).last_hidden_state[:, numeral_p, :].cpu()
                    embeds.append(embed)

                while len(embeds) < args['padding_embed']:
                    embeds.append(empty_embed)
                embeds = torch.cat(embeds, dim=0)
                all_embeds[k] = embeds

            inputs.append(embeds)
            labels.append(v['volatility'][: 29])
            #labels.append([v['volatility'][1], v['volatility'][5], v['volatility'][13]])
    
    if not os.path.exists('train_embed.pickle'):
        with open('train_embed.pickle', 'wb') as fp:
            pickle.dump(all_embeds, fp)
    
    embed_model.to('cpu')
        
    train_dataset = paragraphDataset(inputs, labels)
    
    train_dataloader = DataLoader(
                        train_dataset, 
                        args['batch_size'], 
                        shuffle=True, 
                        collate_fn=collate_fn, 
                        num_workers=4, 
                    )
    return train_dataloader

def get_test_dataloader(args, tokenizer, embed_model):
    with open(args['test_data'], 'r') as fp:
        data = json.load(fp)

    if os.path.exists('test_embed.pickle'):
        with open('test_embed.pickle', 'rb') as fp:
            all_embeds = pickle.load(fp)
    else:
        all_embeds = {}

    inputs, labels = [], []

    embed_model.to('cuda')
    embed_model.eval()
    empty_tokens = tokenizer('', return_tensors='pt', padding='max_length', max_length=args['padding_length'], return_attention_mask=False, return_offsets_mapping=False)
    empty_embed = embed_model(empty_tokens.input_ids.cuda()).last_hidden_state[:, 0, :].cpu()

    with torch.no_grad():
        for k, v in tqdm(data.items(), position = 0, desc='tokenize'):
            sentences = ''
            for sent in v['text']:
                sentences += sent
            try:
                embeds = all_embeds[k]
                while len(embeds) < args['padding_embed']:
                    embeds = torch.cat((embeds, empty_embed), dim=0)
            except KeyError:
                embeds = []
                numeral_idx = get_numeral_idx(sentences)
                all_tokens = tokenizer(sentences, return_tensors='np', return_offsets_mapping=True, return_attention_mask=False, verbose=False)
                for (numeral_l, numeral_r) in numeral_idx:
                    L, R = -1, -1
                    for idx, offset in enumerate(all_tokens.offset_mapping[0]):
                        if offset[0] <= numeral_l and numeral_l < offset[1]:
                            L = idx
                        if offset[0] < numeral_r and numeral_r <= offset[1]:
                            R = idx
                    offset_start, offset_end = -1, -1
                    for idx, offset in enumerate(all_tokens.offset_mapping[0]):
                        if idx >= L - args['extend_token'] and (offset[0] != 0 or offset[1] != 0) and offset_start == -1:
                            offset_start = offset[0]
                        if idx <= R + args['extend_token'] and (offset[0] != 0 or offset[1] != 0):
                            offset_end = offset[1]
                    input_sentence = sentences[offset_start: numeral_l] + ' ' + tokenizer.mask_token + ' ' + sentences[numeral_r: offset_end]
                    input_tokens = tokenizer(input_sentence, return_tensors='pt', padding='max_length', max_length=args['padding_length'], return_attention_mask=False, return_offsets_mapping=False)
                    numeral_p = 0 
                    for idx, token_id in enumerate(input_tokens.input_ids[0]):
                        if token_id == tokenizer.mask_token_id:
                            numeral_p = idx
                            break
                    assert len(input_tokens.input_ids) == 1
                    embed = embed_model(input_tokens.input_ids.cuda()).last_hidden_state[:, numeral_p, :].cpu()
                    embeds.append(embed)

                while len(embeds) < args['padding_embed']:
                    embeds.append(empty_embed)
                embeds = torch.cat(embeds, dim=0)
                all_embeds[k] = embeds

            inputs.append(embeds)
            labels.append(v['volatility'][: 29])
            #labels.append([v['volatility'][1], v['volatility'][5], v['volatility'][13]])
    
    if not os.path.exists('test_embed.pickle'):
        with open('test_embed.pickle', 'wb') as fp:
            pickle.dump(all_embeds, fp)
    
    embed_model.to('cpu')
        
    test_dataset = paragraphDataset(inputs, labels)
    
    test_dataloader = DataLoader(
                        test_dataset, 
                        args['batch_size'], 
                        shuffle=False, 
                        collate_fn=collate_fn, 
                        num_workers=4, 
                    )
    return test_dataloader
