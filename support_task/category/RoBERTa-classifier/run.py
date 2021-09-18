import os
import torch
import random
import numpy as np
from tqdm import tqdm
from model import RobertaNumeralClassifier
from transformers import RobertaTokenizerFast, AdamW
from config import get_config
from data_process import get_train_dataloader, get_eval_dataloader, get_test_dataloader, get_all_category

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
	category = get_all_category(args)

	model = RobertaNumeralClassifier.from_pretrained(args['pretrained_model'], num_labels=len(category))
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
			optimizer.zero_grad()

			inputs, numeral_position, category_id = batch
			logits = model(inputs.cuda(), numeral_position).cpu()
			loss = loss_fn(logits, category_id)
			for logit, label in zip(logits, category_id):
				max_arg = 0
				for idx in range(len(logit)):
					if logit[idx].item() > logit[max_arg].item():
						max_arg = idx
				if max_arg == label.item():
					n_correct += 1
				total += 1
			acc = n_correct / total
			pbar.set_postfix({'acc': '{:.3f}%'.format(acc * 100), 'loss': '{:.3f}'.format(loss.item())})
			step_cnt += 1

			loss.backward()
			optimizer.step()

			if step_cnt % args['saving_steps'] == 0:
				ckpt_dir = os.path.join(args['model_dir'], 'checkpoints')
				if not os.path.exists(ckpt_dir):
					os.makedirs(ckpt_dir)
				ckpt_name = 'ckpt_{}'.format(step_cnt // args['saving_steps'])
				ckpt_path = os.path.join(ckpt_dir, ckpt_name)
				model.save_pretrained(ckpt_path)
	
	model.save_pretrained(save_dir)

def test(args):
	category = get_all_category(args)
	seed_everything(args['seed'])

	tokenizer = RobertaTokenizerFast.from_pretrained(args['pretrained_model'])
	test_dataloader = get_test_dataloader(args, tokenizer)
	model = RobertaNumeralClassifier.from_pretrained(args['testing_model'], num_labels=len(category))

	model.to('cuda')
	model.eval()
	loss_fn = torch.nn.CrossEntropyLoss()
	
	log = []
	n_correct, total = 0, 0
	with torch.no_grad():
		pbar = tqdm(test_dataloader, position=0, desc='inference')
		for batch in pbar:
			inputs, numeral_position, category_id = batch
			logits = model(inputs.cuda(), numeral_position).cpu()
			loss = loss_fn(logits, category_id)
			for logit, label in zip(logits, category_id):
				max_arg = 0
				for idx in range(len(logit)):
					if logit[idx].item() > logit[max_arg].item():
						max_arg = idx
				if max_arg == label.item():
					n_correct += 1
				total += 1
				log.append((category[max_arg], category[label.item()]))
			acc = n_correct / total
			pbar.set_postfix({'acc': '{:.3f}%'.format(acc * 100), 'loss': '{:.3f}'.format(loss.item())})
	
	with open('log/test-log.txt', 'w') as fp:
		for p, r in log:
			fp.write('{} | {}\n'.format(p, r))
			
	log = []
	n_correct, total = 0, 0
	eval_dataloader = get_eval_dataloader(args, tokenizer)
	with torch.no_grad():
		pbar = tqdm(eval_dataloader, position=0, desc='evaluate')
		for batch in pbar:
			inputs, numeral_position, category_id = batch
			logits = model(inputs.cuda(), numeral_position).cpu()
			loss = loss_fn(logits, category_id)
			for logit, label in zip(logits, category_id):
				max_arg = 0
				for idx in range(len(logit)):
					if logit[idx].item() > logit[max_arg].item():
						max_arg = idx
				if max_arg == label.item():
					n_correct += 1
				total += 1
				log.append((category[max_arg], category[label.item()]))
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
