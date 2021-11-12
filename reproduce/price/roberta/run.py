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
from data_process import get_train_dataloader, get_test_dataloader, init_embedding

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

    init_embedding(args, tokenizer, roberta_layer)
    train_dataloader = get_train_dataloader(args)
    test_dataloader = get_test_dataloader(args)

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

    best = 0.0
    best_epoch = -1

    for epoch in range(args['n_epoch']):
        transformer_layer.train()
        linear_layer.train()
        pbar = tqdm(train_dataloader, position=0, desc='Epoch {}'.format(epoch))
        tp, fn, fp, tn = 0, 0, 0, 0

        for batch_embeds, batch_labels in pbar: 
            outputs = transformer_layer(batch_embeds.cuda())
            outputs = torch.squeeze(linear_layer(outputs).cpu(), dim=-1)

            loss = loss_fn(batch_labels, outputs)

            for pred, real in zip(outputs, batch_labels):
                if pred.item() > 0.5:
                    p = 1
                else:
                    p = 0
                if real.item() > 0.5:
                    r = 1
                else:
                    r = 0
                if p == 1 and r == 1:
                    tp += 1
                if p == 1 and r == 0:
                    fp += 1
                if p == 0 and r == 1:
                    fn += 1
                if p == 0 and r == 0:
                    tn += 1

            try:
                prec = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score = 2 * (prec * recall) / (prec + recall)
            except:
                f1_score = 0.0
            pbar.set_postfix({'f1-score': '{:.3f}'.format(f1_score)})
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
                torch.save(transformer_layer, os.path.join(ckpt_path, 'transformer.pt'))
                torch.save(linear_layer, os.path.join(ckpt_path, 'linear.pt'))

        transformer_layer.eval()
        linear_layer.eval()
        with torch.no_grad():
            tp, fn, fp, tn = 0, 0, 0, 0
            for batch_embeds, batch_labels in test_dataloader:
                outputs = transformer_layer(batch_embeds.cuda())
                outputs = torch.squeeze(linear_layer(outputs).cpu(), dim=-1)

                loss = loss_fn(batch_labels, outputs)
                for pred, real in zip(outputs, batch_labels):
                    if pred.item() > 0.5:
                        p = 1
                    else:
                        p = 0
                    if real.item() > 0.5:
                        r = 1
                    else:
                        r = 0
                    if p == 1 and r == 1:
                        tp += 1
                    if p == 1 and r == 0:
                        fp += 1
                    if p == 0 and r == 1:
                        fn += 1
                    if p == 0 and r == 0:
                        tn += 1

            try:
                prec = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score = 2 * (prec * recall) / (prec + recall)
            except:
                f1_score = 0.0
            # print(f1_score)
            if f1_score > best:
                best = f1_score
                best_epoch = epoch
                torch.save(transformer_layer, os.path.join(save_dir, 'transformer_day{}.pt'.format(args['predict_day'])))
                torch.save(linear_layer, os.path.join(save_dir, 'linear_day{}.pt'.format(args['predict_day'])))

        if type(args['saving_epochs']) != type(None) and (epoch + 1) % args['saving_epochs'] == 0:
            ckpt_dir = os.path.join(args['model_dir'], 'checkpoints')
            ckpt_name = 'epoch_ckpt_{}'.format((epoch + 1) // args['saving_epochs'])
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            torch.save(transformer_layer, os.path.join(ckpt_path, 'transformer.pt'))
            torch.save(linear_layer, os.path.join(ckpt_path, 'linear.pt'))
    
    print('Best F1-score: {:.3f}, Best epoch: {}'.format(best, best_epoch))

def test(args):
    seed_everything(args['seed'])

    save_dir = os.path.join(args['model_dir'], 'model')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    roberta_layer = RobertaModel.from_pretrained(args['pretrained_model'])
    transformer_layer = torch.load(os.path.join(save_dir, 'transformer_day{}.pt'.format(args['predict_day'])))
    linear_layer = torch.load(os.path.join(save_dir, 'linear_day{}.pt'.format(args['predict_day'])))
    test_dataloader = get_test_dataloader(args)

    transformer_layer.to('cuda')
    linear_layer.to('cuda') 
    loss_fn = torch.nn.MSELoss()
    transformer_layer.eval()
    linear_layer.eval()
    
    with torch.no_grad():
        tp, fn, fp, tn = 0, 0, 0, 0
        for batch_embeds, batch_labels in test_dataloader:
            outputs = transformer_layer(batch_embeds.cuda())
            outputs = torch.squeeze(linear_layer(outputs).cpu(), dim=-1)

            loss = loss_fn(batch_labels, outputs)
            for pred, real in zip(outputs, batch_labels):
                if pred.item() > 0.5:
                    p = 1
                else:
                    p = 0
                if real.item() > 0.5:
                    r = 1
                else:
                    r = 0
                if p == 1 and r == 1:
                    tp += 1
                if p == 1 and r == 0:
                    fp += 1
                if p == 0 and r == 1:
                    fn += 1
                if p == 0 and r == 0:
                    tn += 1

        try:
            prec = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * (prec * recall) / (prec + recall)
        except:
            f1_score = 0.0
        print(f1_score)

if __name__ == '__main__':
    args = get_config()
    if args['mode'] == 'train':
        train(args)
    if args['mode'] == 'test':
        test(args)
