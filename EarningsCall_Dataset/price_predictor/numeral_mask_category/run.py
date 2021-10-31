import os
import math
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

    #loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.MSELoss()
    
    transformer_layer.train()
    linear_layer.train()

    best = 0
    best_mcc = -1000
    best_epoch = -1

    for epoch in range(args['n_epoch']):
        transformer_layer.train()
        linear_layer.train()
        pbar = tqdm(train_dataloader, position=0, desc='Epoch {}'.format(epoch))
        n_correct, size = 0, 0
        tp, fn, fp, tn = 0, 0, 0, 0

        for batch_embeds, batch_labels in pbar: 
            outputs = transformer_layer(batch_embeds.cuda())
            outputs = torch.squeeze(linear_layer(outputs).cpu())

            # original_labels = [0.95 if label.item() >= 0.9 else 0.05 for label in batch_labels]
            # batch_labels = torch.FloatTensor(original_labels)

            loss = loss_fn(outputs, batch_labels)
            for pred, real in zip(outputs, batch_labels):
                if pred.item() > 0.5:
                    p = 1
                else:
                    p = 0
                if real.item() > 0.5:
                    r = 1
                else:
                    r = 0
                if p == r:
                    n_correct += 1

                if p == 1 and r == 1:
                    tp += 1
                if p == 1 and r == 0:
                    fp += 1
                if p == 0 and r == 1:
                    fn += 1
                if p == 0 and r == 0:
                    tn += 1

                size += 1
            try:
                prec = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score = 2 * (recall * prec) / (recall + prec)
                mcc_score = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            except:
                f1_score = 0.0
                mcc_score = -1000
            pbar.set_postfix({'acc': '{:.3f}'.format(n_correct / size), 'f1_score': '{:.3f}'.format(f1_score), 'mcc_score': '{:.3f}'.format(mcc_score)})
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
            n_correct, size = 0, 0
            tp, fn, fp, tn = 0, 0, 0, 0
            for batch_embeds, batch_labels in test_dataloader:
                outputs = transformer_layer(batch_embeds.cuda())
                outputs = torch.squeeze(linear_layer(outputs).cpu())

                for pred, real in zip(outputs, batch_labels):
                    if pred.item() > 0.5:
                        p = 1
                    else:
                        p = 0
                    if real.item() > 0.5:
                        r = 1
                    else:
                        r = 0
                    if p == r:
                        n_correct += 1

                    if p == 1 and r == 1:
                        tp += 1
                    if p == 1 and r == 0:
                        fp += 1
                    if p == 0 and r == 1:
                        fn += 1
                    if p == 0 and r == 0:
                        tn += 1

                    size += 1
            try:
                prec = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score = 2 * (recall * prec) / (recall + prec)
                mcc_score = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            except:
                f1_score = 0.0
                mcc_score = -1000.0
            pbar.set_postfix({'acc': '{:.3f}'.format(n_correct / size), 'f1_score': '{:.3f}'.format(f1_score), 'mcc_score': '{:.3f}'.format(mcc_score)})
            print(f1_score, mcc_score)
            if f1_score > best:
                best = f1_score
                best_epoch = epoch
            if mcc_score > best_mcc:
                best_mcc = mcc_score

        if type(args['saving_epochs']) != type(None) and (epoch + 1) % args['saving_epochs'] == 0:
            ckpt_dir = os.path.join(args['model_dir'], 'checkpoints')
            ckpt_name = 'epoch_ckpt_{}'.format((epoch + 1) // args['saving_epochs'])
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            roberta_layer.save_pretrained(os.path.join(ckpt_path, 'roberta'))
            torch.save(transformer_layer, os.path.join(ckpt_path, 'transformer.pt'))
            torch.save(linear_layer, os.path.join(ckpt_path, 'linear.pt'))
    
    print('Best f1 score: {:.3f}, Best epoch: {}, Best MCC score: {:.3f}'.format(best, best_epoch, best_mcc))
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
    loss_fn = torch.nn.BCEWithLogitsLoss()
    transformer_layer.eval()
    linear_layer.eval()
    
    with torch.no_grad():
        pbar = tqdm(test_dataloader, position=0, desc='inference')
        n_correct, size = 0, 0
        for batch_embeds, batch_labels in pbar:
            outputs = transformer_layer(batch_embeds.cuda())
            outputs = torch.squeeze(linear_layer(outputs).cpu())

            loss = loss_fn(outputs, batch_labels)
            
            for pred, real in zip(outputs, batch_labels):
                if pred.item() > 0.5:
                    p = 1
                else:
                    p = 0
                if real.item() > 0.5:
                    r = 1
                else:
                    r = 0
                if p == r:
                    n_correct += 1
                size += 1

            pbar.set_postfix({'loss': '{:.3f}'.format(loss.item()), 'acc': '{:.3f}'.format(n_correct / size)})

if __name__ == '__main__':
    args = get_config()
    if args['mode'] == 'train':
        train(args)
    if args['mode'] == 'test':
        test(args)
