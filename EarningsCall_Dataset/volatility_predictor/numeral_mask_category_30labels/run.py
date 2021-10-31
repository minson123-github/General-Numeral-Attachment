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
from data_process import get_train_dataloader, get_test_dataloader

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
    transformer_layer = TransformerLayer(embed_dim, 1024, 16, 0)
    linear_layer = LinearLayer(1024, 256)

    transformer_layer.to('cuda')
    linear_layer.to('cuda')

    train_dataloader = get_train_dataloader(args, tokenizer, roberta_layer)
    test_dataloader = get_test_dataloader(args, tokenizer, roberta_layer)

    save_dir = os.path.join(args['model_dir'], 'model')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    step_cnt = 0

    optimizer = AdamW(
        itertools.chain(
            transformer_layer.parameters(), 
            linear_layer.parameters()
        ), 
        lr=args['lr']
    )

    loss_fn = torch.nn.MSELoss()
    
    transformer_layer.train()
    linear_layer.train()

    best = 1000
    best_epoch = -1
    best_triple = (0, 0, 0)

    for epoch in range(args['n_epoch']):
        transformer_layer.train()
        linear_layer.train()
        pbar = tqdm(train_dataloader, position=0, desc='Epoch {}'.format(epoch))
        total, size = 0, 0

        for batch_embeds, batch_labels in pbar: 
            outputs = transformer_layer(batch_embeds.cuda())
            outputs = linear_layer(outputs).cpu()

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

        transformer_layer.eval()
        linear_layer.eval()
        with torch.no_grad():
            r3_total, r3_size = 0, 0
            r7_total, r7_size = 0, 0
            r15_total, r15_size = 0, 0
            for batch_embeds, batch_labels in test_dataloader:
                outputs = transformer_layer(batch_embeds.cuda())
                outputs = linear_layer(outputs).cpu()
                r3_outputs = outputs[:, 1]
                r3_batch_labels = batch_labels[:, 1]
                loss = loss_fn(r3_batch_labels, r3_outputs)
                r3_total += loss.item()
                r3_size += 1

                r7_outputs = outputs[:, 5]
                r7_batch_labels = batch_labels[:, 5]
                loss = loss_fn(r7_batch_labels, r7_outputs)
                r7_total += loss.item()
                r7_size += 1

                r15_outputs = outputs[:, 13]
                r15_batch_labels = batch_labels[:, 13]
                loss = loss_fn(r15_batch_labels, r15_outputs)
                r15_total += loss.item()
                r15_size += 1
            print(r3_total / r3_size, r7_total / r7_size, r15_total / r15_size)
            if (r3_total / r3_size + r7_total / r7_size + r15_total / r15_size) / 3 < best:
                best = (r3_total / r3_size + r7_total / r7_size + r15_total / r15_size) / 3
                best_triple = (r3_total / r3_size, r7_total / r7_size, r15_total / r15_size)
                best_epoch = epoch

        if type(args['saving_epochs']) != type(None) and (epoch + 1) % args['saving_epochs'] == 0:
            ckpt_dir = os.path.join(args['model_dir'], 'checkpoints')
            ckpt_name = 'epoch_ckpt_{}'.format((epoch + 1) // args['saving_epochs'])
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            roberta_layer.save_pretrained(os.path.join(ckpt_path, 'roberta'))
            torch.save(transformer_layer, os.path.join(ckpt_path, 'transformer.pt'))
            torch.save(linear_layer, os.path.join(ckpt_path, 'linear.pt'))
    
    print('Best test mse: {:.3f}, {:.3f}, {:.3f}, Best epoch: {}'.format(best_triple[0], best_triple[1], best_triple[2], best_epoch))
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
            outputs = outputs[:, 1] + outputs[:, 5] + outputs[:, 13]
            batch_labels = batch_labels[:, 1] + batch_labels[:, 5] + batch_labels[:, 13]

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
