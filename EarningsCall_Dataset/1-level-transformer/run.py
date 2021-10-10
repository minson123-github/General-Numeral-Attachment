import os
import time
import torch
import random
import itertools
import numpy as np
from tqdm import tqdm
from model import TransformerLayer, LinearLayer
from transformers import RobertaModel, RobertaTokenizerFast, AdamW
from config import get_config
from data_process import get_train_dataloader, get_eval_dataloader, get_test_dataloader

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def seed_everything(seed):
    print('Setting global seed to {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(args):
    seed_everything(args['seed'])
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')

    roberta_layer = RobertaModel.from_pretrained(args['pretrained_model'])
    embed_dim = roberta_layer.config.hidden_size
    transformer_layer = TransformerLayer(embed_dim, 8, 1024, 0.1)
    linear_layer = LinearLayer(embed_dim, 256)

    transformer_layer.to('cuda')
    linear_layer.to('cuda')

    train_dataloader = get_train_dataloader(args, tokenizer, roberta_layer)

    save_dir = os.path.join(args['model_dir'], 'model')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    step_cnt = 0

    optimizer = AdamW(
        itertools.chain(
            roberta_layer.parameters(), 
            transformer_layer.parameters(), 
            linear_layer.parameters()
        ), 
        lr=args['lr']
    )

    loss_fn = torch.nn.MSELoss()
    
    transformer_layer.train()
    linear_layer.train()

    for epoch in range(args['n_epoch']):
        pbar = tqdm(train_dataloader, position=0, desc='Epoch {}'.format(epoch))
        total, size = 0, 0

        for batch_embeds, batch_labels in pbar: 
            outputs = transformer_layer(batch_embeds.cuda())
            outputs = torch.squeeze(linear_layer(outputs).cpu())

            loss = loss_fn(batch_labels, outputs)
            total += loss.item()
            size += 1
            pbar.set_postfix({'mse': '{:.3f}'.format(total / size)})
            step_cnt += 1

            loss.backward()

            if step_cnt % args['accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            if type(args['saving_steps']) != type(None) and step_cnt % args['saving_steps'] == 0:
                ckpt_dir = os.path.join(args['model_dir'], 'checkpoints')
                ckpt_name = 'step_ckpt_{}'.format(step_cnt // args['saving_steps'])
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
                roberta_layer.save_pretrained(os.path.join(ckpt_path, 'roberta'))
                torch.save(transformer_layer, os.path.join(ckpt_path, 'transformer.pt'))
                torch.save(linear_layer, os.path.join(ckpt_path, 'linear.pt'))

        if type(args['saving_epochs']) != type(None) and (epoch + 1) % args['saving_epochs'] == 0:
            ckpt_dir = os.path.join(args['model_dir'], 'checkpoints')
            ckpt_name = 'epoch_ckpt_{}'.format((epoch + 1) // args['saving_epochs'])
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            roberta_layer.save_pretrained(os.path.join(ckpt_path, 'roberta'))
            torch.save(transformer_layer, os.path.join(ckpt_path, 'transformer.pt'))
            torch.save(linear_layer, os.path.join(ckpt_path, 'linear.pt'))
    
    roberta_layer.save_pretrained(os.path.join(save_dir, 'roberta'))
    torch.save(transformer_layer, os.path.join(save_dir, 'transformer.pt'))
    torch.save(linear_layer, os.path.join(save_dir, 'linear.pt'))

def test(args):
    seed_everything(args['seed'])

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    roberta_layer = RobertaModel.from_pretrained(os.path.join(args['testing_model'], 'roberta'))
    transformer_layer = torch.load(os.path.join(args['testing_model'], 'transformer.pt'))
    linear_layer = torch.load(os.path.join(args['testing_model'], 'linear.pt'))
    test_dataloader = get_test_dataloader(args, tokenizer, roberta_layer)

    transformer_layer.to('cuda')
    linear_layer.to('cuda') 
    loss_fn = torch.nn.MSELoss()
    transformer_layer.eval()
    linear_layer.eval()
    
    with torch.no_grad():
        pbar = tqdm(test_dataloader, position=0, desc='inference')
        total, size = 0, 0
        for batch_embeds, batch_labels in pbar:
            outputs = transformer_layer(batch_embeds.cuda())
            outputs = torch.squeeze(linear_layer(outputs).cpu())

            loss = loss_fn(batch_labels, outputs)
            total += loss.item()
            size += 1
            pbar.set_postfix({'mse': '{:.3f}'.format(total / size)})

if __name__ == '__main__':
    args = get_config()
    if args['mode'] == 'train':
        train(args)
    if args['mode'] == 'test':
        test(args)
