import os
import time
import torch
import random
import itertools
import numpy as np
from tqdm import tqdm
from model import TransformerLayer, seq2seqLayer
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
    seq2seq_layer = seq2seqLayer(1024, 256)

    transformer_layer.to('cuda')
    seq2seq_layer.to('cuda')

    train_dataloader = get_train_dataloader(args, tokenizer, roberta_layer)
    test_dataloader = get_test_dataloader(args, tokenizer, roberta_layer)

    save_dir = os.path.join(args['model_dir'], 'model')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    step_cnt = 0

    optimizer = AdamW(
        itertools.chain(
            transformer_layer.parameters(), 
            seq2seq_layer.parameters()
        ), 
        lr=args['lr']
    )

    loss_fn = torch.nn.MSELoss()
    
    transformer_layer.train()
    seq2seq_layer.train()

    best = 0
    best_epoch = -1
    best_triple = (0, 0, 0, 0, 0)
    seq_len = 30

    for epoch in range(args['n_epoch']):
        transformer_layer.train()
        seq2seq_layer.train()
        pbar = tqdm(train_dataloader, position=0, desc='Epoch {}'.format(epoch))

        for batch_embeds, batch_labels in pbar: 
            transformer_outputs = transformer_layer(batch_embeds.cuda())
            hidden = None
            seq2seq_outputs = []
            for i in range(seq_len):
                outputs, hidden = seq2seq_layer(transformer_outputs, hidden)
                seq2seq_outputs.append(torch.squeeze(outputs, dim=2))
            seq2seq_outputs = torch.cat(seq2seq_outputs, dim=1).cpu()

            loss = loss_fn(batch_labels, seq2seq_outputs)
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
        seq2seq_layer.eval()
        with torch.no_grad():
            r1_tp, r1_fn, r1_fp, r1_tn = 0, 0, 0, 0
            r3_tp, r3_fn, r3_fp, r3_tn = 0, 0, 0, 0
            r7_tp, r7_fn, r7_fp, r7_tn = 0, 0, 0, 0
            r15_tp, r15_fn, r15_fp, r15_tn = 0, 0, 0, 0
            r30_tp, r30_fn, r30_fp, r30_tn = 0, 0, 0, 0
            for batch_embeds, batch_labels in test_dataloader:
                transformer_outputs = transformer_layer(batch_embeds.cuda())
                hidden = None
                seq2seq_outputs = []
                for i in range(seq_len):
                    outputs, hidden = seq2seq_layer(transformer_outputs, hidden)
                    seq2seq_outputs.append(torch.squeeze(outputs, dim=2))
                seq2seq_outputs = torch.cat(seq2seq_outputs, dim=1).cpu()
                outputs = seq2seq_outputs

                r1_outputs = outputs[:, 0]
                r1_batch_labels = batch_labels[:, 0]
                for pred, real in zip(r1_outputs, r1_batch_labels):
                    if pred.item() > 0.5:
                        p = 1
                    else:
                        p = 0
                    if real.item() > 0.5:
                        r = 1
                    else:
                        r = 0
                    if p == 1 and r == 1:
                        r1_tp += 1
                    if p == 1 and r == 0:
                        r1_fp += 1
                    if p == 0 and r == 1:
                        r1_fn += 1
                    if p == 0 and r == 0:
                        r1_tn += 1

                try:
                    r1_prec = r1_tp / (r1_tp + r1_fp)
                    r1_recall = r1_tp / (r1_tp + r1_fn)
                    r1_f1_score = 2 * (r1_prec * r1_recall) / (r1_prec + r1_recall)
                except:
                    r1_f1_score = 0.0

                r3_outputs = outputs[:, 2]
                r3_batch_labels = batch_labels[:, 2]
                for pred, real in zip(r3_outputs, r3_batch_labels):
                    if pred.item() > 0.5:
                        p = 1
                    else:
                        p = 0
                    if real.item() > 0.5:
                        r = 1
                    else:
                        r = 0
                    if p == 1 and r == 1:
                        r3_tp += 1
                    if p == 1 and r == 0:
                        r3_fp += 1
                    if p == 0 and r == 1:
                        r3_fn += 1
                    if p == 0 and r == 0:
                        r3_tn += 1

                try:
                    r3_prec = r3_tp / (r3_tp + r3_fp)
                    r3_recall = r3_tp / (r3_tp + r3_fn)
                    r3_f1_score = 2 * (r3_prec * r3_recall) / (r3_prec + r3_recall)
                except:
                    r3_f1_score = 0.0

                r7_outputs = outputs[:, 6]
                r7_batch_labels = batch_labels[:, 6]
                for pred, real in zip(r7_outputs, r7_batch_labels):
                    if pred.item() > 0.5:
                        p = 1
                    else:
                        p = 0
                    if real.item() > 0.5:
                        r = 1
                    else:
                        r = 0
                    if p == 1 and r == 1:
                        r7_tp += 1
                    if p == 1 and r == 0:
                        r7_fp += 1
                    if p == 0 and r == 1:
                        r7_fn += 1
                    if p == 0 and r == 0:
                        r7_tn += 1

                try:
                    r7_prec = r7_tp / (r7_tp + r7_fp)
                    r7_recall = r7_tp / (r7_tp + r7_fn)
                    r7_f1_score = 2 * (r7_prec * r7_recall) / (r7_prec + r7_recall)
                except:
                    r7_f1_score = 0.0

                r15_outputs = outputs[:, 14]
                r15_batch_labels = batch_labels[:, 14]
                for pred, real in zip(r15_outputs, r15_batch_labels):
                    if pred.item() > 0.5:
                        p = 1
                    else:
                        p = 0
                    if real.item() > 0.5:
                        r = 1
                    else:
                        r = 0
                    if p == 1 and r == 1:
                        r15_tp += 1
                    if p == 1 and r == 0:
                        r15_fp += 1
                    if p == 0 and r == 1:
                        r15_fn += 1
                    if p == 0 and r == 0:
                        r15_tn += 1

                try:
                    r15_prec = r15_tp / (r15_tp + r15_fp)
                    r15_recall = r15_tp / (r15_tp + r15_fn)
                    r15_f1_score = 2 * (r15_prec * r15_recall) / (r15_prec + r15_recall)
                except:
                    r15_f1_score = 0.0

                r30_outputs = outputs[:, 29]
                r30_batch_labels = batch_labels[:, 29]
                for pred, real in zip(r30_outputs, r30_batch_labels):
                    if pred.item() > 0.5:
                        p = 1
                    else:
                        p = 0
                    if real.item() > 0.5:
                        r = 1
                    else:
                        r = 0
                    if p == 1 and r == 1:
                        r30_tp += 1
                    if p == 1 and r == 0:
                        r30_fp += 1
                    if p == 0 and r == 1:
                        r30_fn += 1
                    if p == 0 and r == 0:
                        r30_tn += 1

                try:
                    r30_prec = r30_tp / (r30_tp + r30_fp)
                    r30_recall = r30_tp / (r30_tp + r30_fn)
                    r30_f1_score = 2 * (r30_prec * r30_recall) / (r30_prec + r30_recall)
                except:
                    r30_f1_score = 0.0

            if (r1_f1_score + r3_f1_score + r7_f1_score + r15_f1_score + r30_f1_score) > best:
                best = (r1_f1_score + r3_f1_score + r7_f1_score + r15_f1_score + r30_f1_score)
                best_triple = (r1_f1_score, r3_f1_score, r7_f1_score, r15_f1_score, r30_f1_score)
                best_epoch = epoch
                torch.save(transformer_layer, os.path.join(save_dir, 'transformer.pt'))
                torch.save(seq2seq_layer, os.path.join(save_dir, 'lstm.pt'))

        if type(args['saving_epochs']) != type(None) and (epoch + 1) % args['saving_epochs'] == 0:
            ckpt_dir = os.path.join(args['model_dir'], 'checkpoints')
            ckpt_name = 'epoch_ckpt_{}'.format((epoch + 1) // args['saving_epochs'])
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            roberta_layer.save_pretrained(os.path.join(ckpt_path, 'roberta'))
            torch.save(transformer_layer, os.path.join(ckpt_path, 'transformer.pt'))
            torch.save(linear_layer, os.path.join(ckpt_path, 'linear.pt'))
    
    print('Best test mse: {:.3f} {:.3f}, {:.3f}, {:.3f}, {:.3f}, Best epoch: {}'.format(best_triple[0], best_triple[1], best_triple[2], best_triple[3], best_triple[4], best_epoch))

def test(args):
    seed_everything(args['seed'])

    save_dir = os.path.join(args['model_dir'], 'model')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    roberta_layer = RobertaModel.from_pretrained(args['pretrained_model'])
    transformer_layer = torch.load(os.path.join(save_dir, 'transformer.pt'))
    seq2seq_layer = torch.load(os.path.join(save_dir, 'lstm.pt'))
    test_dataloader = get_test_dataloader(args, tokenizer, roberta_layer)

    transformer_layer.to('cuda')
    seq2seq_layer.to('cuda') 
    loss_fn = torch.nn.MSELoss()
    transformer_layer.eval()
    seq2seq_layer.eval()
    seq_len = 30

    with torch.no_grad():
        r1_tp, r1_fn, r1_fp, r1_tn = 0, 0, 0, 0
        r3_tp, r3_fn, r3_fp, r3_tn = 0, 0, 0, 0
        r7_tp, r7_fn, r7_fp, r7_tn = 0, 0, 0, 0
        r15_tp, r15_fn, r15_fp, r15_tn = 0, 0, 0, 0
        r30_tp, r30_fn, r30_fp, r30_tn = 0, 0, 0, 0
        for batch_embeds, batch_labels in test_dataloader:
            transformer_outputs = transformer_layer(batch_embeds.cuda())
            hidden = None
            seq2seq_outputs = []
            for i in range(seq_len):
                outputs, hidden = seq2seq_layer(transformer_outputs, hidden)
                seq2seq_outputs.append(torch.squeeze(outputs, dim=2))
            seq2seq_outputs = torch.cat(seq2seq_outputs, dim=1).cpu()
            outputs = seq2seq_outputs

            r1_outputs = outputs[:, 0]
            r1_batch_labels = batch_labels[:, 0]
            for pred, real in zip(r1_outputs, r1_batch_labels):
                if pred.item() > 0.5:
                    p = 1
                else:
                    p = 0
                if real.item() > 0.5:
                    r = 1
                else:
                    r = 0
                if p == 1 and r == 1:
                    r1_tp += 1
                if p == 1 and r == 0:
                    r1_fp += 1
                if p == 0 and r == 1:
                    r1_fn += 1
                if p == 0 and r == 0:
                    r1_tn += 1

            try:
                r1_prec = r1_tp / (r1_tp + r1_fp)
                r1_recall = r1_tp / (r1_tp + r1_fn)
                r1_f1_score = 2 * (r1_prec * r1_recall) / (r1_prec + r1_recall)
            except:
                r1_f1_score = 0.0

            r3_outputs = outputs[:, 2]
            r3_batch_labels = batch_labels[:, 2]
            for pred, real in zip(r3_outputs, r3_batch_labels):
                if pred.item() > 0.5:
                    p = 1
                else:
                    p = 0
                if real.item() > 0.5:
                    r = 1
                else:
                    r = 0
                if p == 1 and r == 1:
                    r3_tp += 1
                if p == 1 and r == 0:
                    r3_fp += 1
                if p == 0 and r == 1:
                    r3_fn += 1
                if p == 0 and r == 0:
                    r3_tn += 1

            try:
                r3_prec = r3_tp / (r3_tp + r3_fp)
                r3_recall = r3_tp / (r3_tp + r3_fn)
                r3_f1_score = 2 * (r3_prec * r3_recall) / (r3_prec + r3_recall)
            except:
                r3_f1_score = 0.0

            r7_outputs = outputs[:, 6]
            r7_batch_labels = batch_labels[:, 6]
            for pred, real in zip(r7_outputs, r7_batch_labels):
                if pred.item() > 0.5:
                    p = 1
                else:
                    p = 0
                if real.item() > 0.5:
                    r = 1
                else:
                    r = 0
                if p == 1 and r == 1:
                    r7_tp += 1
                if p == 1 and r == 0:
                    r7_fp += 1
                if p == 0 and r == 1:
                    r7_fn += 1
                if p == 0 and r == 0:
                    r7_tn += 1

            try:
                r7_prec = r7_tp / (r7_tp + r7_fp)
                r7_recall = r7_tp / (r7_tp + r7_fn)
                r7_f1_score = 2 * (r7_prec * r7_recall) / (r7_prec + r7_recall)
            except:
                r7_f1_score = 0.0

            r15_outputs = outputs[:, 14]
            r15_batch_labels = batch_labels[:, 14]
            for pred, real in zip(r15_outputs, r15_batch_labels):
                if pred.item() > 0.5:
                    p = 1
                else:
                    p = 0
                if real.item() > 0.5:
                    r = 1
                else:
                    r = 0
                if p == 1 and r == 1:
                    r15_tp += 1
                if p == 1 and r == 0:
                    r15_fp += 1
                if p == 0 and r == 1:
                    r15_fn += 1
                if p == 0 and r == 0:
                    r15_tn += 1

            try:
                r15_prec = r15_tp / (r15_tp + r15_fp)
                r15_recall = r15_tp / (r15_tp + r15_fn)
                r15_f1_score = 2 * (r15_prec * r15_recall) / (r15_prec + r15_recall)
            except:
                r15_f1_score = 0.0

            r30_outputs = outputs[:, 29]
            r30_batch_labels = batch_labels[:, 29]
            for pred, real in zip(r30_outputs, r30_batch_labels):
                if pred.item() > 0.5:
                    p = 1
                else:
                    p = 0
                if real.item() > 0.5:
                    r = 1
                else:
                    r = 0
                if p == 1 and r == 1:
                    r30_tp += 1
                if p == 1 and r == 0:
                    r30_fp += 1
                if p == 0 and r == 1:
                    r30_fn += 1
                if p == 0 and r == 0:
                    r30_tn += 1

            try:
                r30_prec = r30_tp / (r30_tp + r30_fp)
                r30_recall = r30_tp / (r30_tp + r30_fn)
                r30_f1_score = 2 * (r30_prec * r30_recall) / (r30_prec + r30_recall)
            except:
                r30_f1_score = 0.0
    print(r1_f1_score, r3_f1_score, r7_f1_score, r15_f1_score, r30_f1_score)

if __name__ == '__main__':
    args = get_config()
    if args['mode'] == 'train':
        train(args)
    if args['mode'] == 'test':
        test(args)
