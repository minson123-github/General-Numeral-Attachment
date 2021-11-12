import os
import json
import torch
import random
import pickle
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

def get_embedding(input_ids, model):
    embeds = []
    batch_size = 256
    for batch_idx in range((len(input_ids) + batch_size - 1) // batch_size):
        batch_ids = input_ids[batch_idx * batch_size: min(len(input_ids), (batch_idx + 1) * batch_size)]
        batch_embed = model(batch_ids.cuda()).last_hidden_state[:, 0, :].cpu()
        embeds.append(batch_embed)
    embeds = torch.cat(embeds, dim=0)
    return embeds

def init_embedding(args, tokenizer, embed_model):
    if not os.path.exists('embed.pickle'):
        with open(args['data'], 'r') as fp:
            data = json.load(fp)

        embed_model.to('cuda')
        embed_model.eval()

        max_num_sent = 0
        for k, v in data.items():
            max_num_sent = max(max_num_sent, len(v['text']))

        all_embeds = {}

        with torch.no_grad():
            for k, v in tqdm(data.items(), position = 0, desc='tokenize'):
                sentences = [sent[:-1] for sent in v['text']]
                while len(sentences) < max_num_sent:
                    sentences.append('')
                try:
                    input_tokens = tokenizer(sentences, return_tensors='pt', padding='max_length', max_length=args['padding_length'], return_attention_mask=False, verbose=False)
                except:
                    continue

                embeds = get_embedding(input_tokens.input_ids, embed_model)
                all_embeds[k] = embeds

        embed_model.to('cpu')

        with open('embed.pickle', 'wb') as fp:
            pickle.dump(all_embeds, fp)

def get_train_dataloader(args):
    with open(args['train_data'], 'r') as fp:
        data = json.load(fp)
    with open('embed.pickle', 'rb') as fp:
        all_embeds = pickle.load(fp)
    inputs, labels = [], []

    for k, v in tqdm(data.items(), position = 0, desc='tokenize'):
        sentences = [sent[:-1] for sent in v['text']]
        try:
            embeds = all_embeds[k]
        except:
            continue

        inputs.append(embeds)
        labels.append(v['volatility'][args['predict_day'] - 2])
        
    train_dataset = paragraphDataset(inputs, labels)
    
    train_dataloader = DataLoader(
                        train_dataset, 
                        args['batch_size'], 
                        shuffle=True, 
                        collate_fn=collate_fn, 
                        num_workers=4, 
                    )
    return train_dataloader

def get_test_dataloader(args):
    with open(args['test_data'], 'r') as fp:
        data = json.load(fp)
    with open('embed.pickle', 'rb') as fp:
        all_embeds = pickle.load(fp)
    inputs, labels = [], []

    for k, v in tqdm(data.items(), position = 0, desc='tokenize'):
        sentences = [sent[:-1] for sent in v['text']]
        try:
            embeds = all_embeds[k]
        except:
            continue

        inputs.append(embeds)
        labels.append(v['volatility'][args['predict_day'] - 2])
        
    test_dataset = paragraphDataset(inputs, labels)

    test_dataloader = DataLoader(
                        test_dataset, 
                        args['batch_size'], 
                        shuffle=False, 
                        collate_fn=collate_fn, 
                        num_workers=4, 
                    )
    return test_dataloader
