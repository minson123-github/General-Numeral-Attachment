import os
import torch
import random
import numpy as np
from tqdm import tqdm
from model import RobertaNumeralQA
from transformers import RobertaTokenizerFast, AdamW
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
	tokenizer = RobertaTokenizerFast.from_pretrained(args['pretrained_model'])

	model = RobertaNumeralQA.from_pretrained(args['pretrained_model'])
	model.to(torch.device("cuda"))

	train_dataloader = get_train_dataloader(args, tokenizer)

	save_dir = os.path.join(args['model_dir'], 'model')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	
	step_cnt = 0

	optimizer = AdamW(model.parameters(), lr=args['lr'])

	loss_fn = torch.nn.CrossEntropyLoss()
	
	model.train()
	for epoch in range(args['n_epoch']):
		pbar = tqdm(train_dataloader, position=0, desc='Epoch {}'.format(epoch))
		n_correct, total = 0, 0
		for batch in pbar:

			inputs, numeral_position, start_position, end_position = batch
			start_logits, end_logits = model(inputs.cuda(), numeral_position)
			start_logits = start_logits.cpu()
			end_logits = end_logits.cpu()
			start_loss = loss_fn(start_logits, start_position)
			end_loss = loss_fn(end_logits, end_position)
			loss = (start_loss + end_loss) / 2
			loss = loss / args['accumulation_steps']
			start_pred = torch.argmax(start_logits, dim=-1)
			end_pred = torch.argmax(end_logits, dim=-1)
			for start_p, end_p, start_r, end_r, input_ids in zip(start_pred, end_pred, start_position, end_position, inputs):
				if start_p >= end_p:
					p = 'None'
				else:
					p = tokenizer.decode(input_ids[start_p: end_p])
				
				if start_r >= end_r:
					r = 'None'
				else:
					r = tokenizer.decode(input_ids[start_r: end_r])
				if p == r:
					n_correct += 1
				total += 1

			acc = n_correct / total
			pbar.set_postfix({'acc': '{:.3f}%'.format(acc * 100), 'loss': '{:.3f}'.format(loss.item())})
			step_cnt += 1

			loss.backward()

			if step_cnt % args['accumulation_steps'] == 0:
				optimizer.step()
				optimizer.zero_grad()


			if type(args['saving_steps']) != type(None) and step_cnt % args['saving_steps'] == 0:
				ckpt_dir = os.path.join(args['model_dir'], 'checkpoints')
				if not os.path.exists(ckpt_dir):
					os.makedirs(ckpt_dir)
				ckpt_name = 'step_ckpt_{}'.format(step_cnt // args['saving_steps'])
				ckpt_path = os.path.join(ckpt_dir, ckpt_name)
				model.save_pretrained(ckpt_path)

		if type(args['saving_epochs']) != type(None) and (epoch + 1) % args['saving_epochs'] == 0:
			ckpt_dir = os.path.join(args['model_dir'], 'checkpoints')
			if not os.path.exists(ckpt_dir):
				os.makedirs(ckpt_dir)
			ckpt_name = 'epoch_ckpt_{}'.format(epoch // args['saving_epochs'])
			ckpt_path = os.path.join(ckpt_dir, ckpt_name)
			model.save_pretrained(ckpt_path)
	
	model.save_pretrained(save_dir)

def test(args):
	seed_everything(args['seed'])

	tokenizer = RobertaTokenizerFast.from_pretrained(args['pretrained_model'])
	test_dataloader = get_test_dataloader(args, tokenizer)
	model = RobertaNumeralQA.from_pretrained(args['testing_model'])

	model.to('cuda')
	model.eval()
	loss_fn = torch.nn.CrossEntropyLoss()
	
	log = []
	n_correct, total = 0, 0
	with torch.no_grad():
		pbar = tqdm(test_dataloader, position=0, desc='inference')
		for batch in pbar:
			inputs, numeral_position, start_position, end_position = batch
			start_logits, end_logits = model(inputs.cuda(), numeral_position)
			start_logits = start_logits.cpu()
			end_logits = end_logits.cpu()
			start_loss = loss_fn(start_logits, start_position)
			end_loss = loss_fn(end_logits, end_position)
			loss = (start_loss + end_loss) / 2
			start_pred = torch.argmax(start_logits, dim=-1)
			end_pred = torch.argmax(end_logits, dim=-1)
			for start_p, end_p, start_r, end_r, input_ids in zip(start_pred, end_pred, start_position, end_position, inputs):
				if start_p >= end_p:
					p = 'None'
				else:
					p = tokenizer.decode(input_ids[start_p: end_p])
				
				if start_r >= end_r:
					r = 'None'
				else:
					r = tokenizer.decode(input_ids[start_r: end_r])
				if p == r:
					n_correct += 1
				log.append((p, r))

				total += 1
			acc = n_correct / total
			pbar.set_postfix({'acc': '{:.3f}%'.format(acc * 100), 'loss': '{:.3f}'.format(loss.item())})
	
	
	with open('log/test-log.txt', 'w') as fp:
		for p, r in log:
			fp.write('{} | {}\n'.format(p, r))
			
	eval_dataloader = get_eval_dataloader(args, tokenizer)
	log = []
	n_correct, total = 0, 0
	with torch.no_grad():
		pbar = tqdm(eval_dataloader, position=0, desc='evaluate')
		for batch in pbar:
			inputs, numeral_position, start_position, end_position = batch
			start_logits, end_logits = model(inputs.cuda(), numeral_position)
			start_logits = start_logits.cpu()
			end_logits = end_logits.cpu()
			start_loss = loss_fn(start_logits, start_position)
			end_loss = loss_fn(end_logits, end_position)
			loss = (start_loss + end_loss) / 2
			start_pred = torch.argmax(start_logits, dim=-1)
			end_pred = torch.argmax(end_logits, dim=-1)
			for start_p, end_p, start_r, end_r, input_ids in zip(start_pred, end_pred, start_position, end_position, inputs):
				if start_p >= end_p:
					p = 'None'
				else:
					p = tokenizer.decode(input_ids[start_p: end_p])
				
				if start_r >= end_r:
					r = 'None'
				else:
					r = tokenizer.decode(input_ids[start_r: end_r])
				if p == r:
					n_correct += 1
				log.append((p, r))

				total += 1
			acc = n_correct / total
			pbar.set_postfix({'acc': '{:.3f}%'.format(acc * 100), 'loss': '{:.3f}'.format(loss.item())})
	
	with open('log/eval-log.txt', 'w') as fp:
		for p, r in log:
			fp.write('{} | {}\n'.format(p, r))

if __name__ == '__main__':
	args = get_config()
	if args['mode'] == 'train':
		train(args)
	if args['mode'] == 'test':
		test(args)
