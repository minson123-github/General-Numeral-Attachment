import os
import torch
from tqdm import tqdm
from model import RobertaClassifier
from transformers import RobertaTokenizerFast
from config import get_config
from pytorch_lightning import Trainer, seed_everything, Callback
from data_process import get_train_dataloader, get_eval_dataloader, get_test_dataloader, get_all_name_entity

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class CheckpointEveryNSteps(Callback):
	def __init__(self, save_freq, save_dir):
		self.save_dir = save_dir
		self.save_freq = save_freq
		self.save_cnt = 0
	
	def on_batch_end(self, trainer: Trainer, _):
		global_step = trainer.global_step
		if (global_step + 1) % self.save_freq == 0:
			self.save_cnt += 1
			filename = 'checkpoint_{}.ckpt'.format(self.save_cnt)
			ckpt_path = os.path.join(self.save_dir, filename)
			trainer.save_checkpoint(ckpt_path)

def train(args):
	seed_everything(args['seed'], workers=True)
	tokenizer = RobertaTokenizerFast.from_pretrained(args['pretrained_model'])
	name_entity = get_all_name_entity(args)
	model = RobertaClassifier(args['pretrained_model'], len(name_entity), args['lr'])

	train_dataloader = get_train_dataloader(args, tokenizer)

	save_path = os.path.join(args['model_dir'], 'model')
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	callbacks = []
	if type(args['saving_steps']) != type(None):
		callbacks = [CheckpointEveryNSteps(args['saving_steps'], os.path.join(args['model_dir'], 'checkpoints'))]
		if not os.path.exists(os.path.join(args['model_dir'], 'checkpoints')):
			os.makedirs(os.path.join(args['model_dir'], 'checkpoints'))
	
	trainer = Trainer(
					deterministic=True, 
					gpus=args['n_gpu'], 
					default_root_dir=save_path, 
					max_epochs=args['n_epoch'], 
					num_nodes=1, 
					# precision=16, 
					accelerator="ddp", 
					# amp_backend='apex', 
					# plugins='deepspeed_stage_2', 
					plugins='ddp_sharded', 
					# accumulate_grad_batches=16, 
					# plugins=DeepSpeedPlugin(deepspeed_config), 
					callbacks = callbacks, 
				)

	if args['auto_lr']:
		trainer = Trainer(
						deterministic=True, 
						gpus=1, 
						default_root_dir=save_path, 
						max_epochs=args['n_epoch'], 
						num_nodes=1, 
					)
		lr_finder = trainer.tuner.lr_find(model, train_dataloader)
		print(lr_finder.results)
		new_lr = lr_finder.suggestion()
		print('The best learning rate is:', new_lr)
		model.hparams.lr = new_lr
	
	print('The learning rate be used for training:', model.lr, flush=True)
	trainer.fit(model, train_dataloader)
	model.model.save_pretrained(save_path)

def test(args):
	name_entity = get_all_name_entity(args)
	seed_everything(args['seed'], workers=True)
	model = RobertaClassifier.load_from_checkpoint(args['testing_model'])
	model.to('cuda')
	tokenizer = RobertaTokenizerFast.from_pretrained(args['pretrained_model'])
	test_dataloader = get_test_dataloader(args, tokenizer)
	n_correct = 0
	size = 0
	log = []
	model.eval()
	with torch.no_grad():
		for (inputs, labels) in tqdm(test_dataloader, position=0, desc='inference'):
			outputs = model.model(inputs.cuda())
			pred = []
			for logit in outputs.logits:
				max_p = 0
				for i in range(len(logit)):
					if logit[i] > logit[max_p]:
						max_p = i
				pred.append(max_p)
			real = [label for label in labels]

			for p, r in zip(pred, real):
				if p == r:
					n_correct += 1
				log.append((name_entity[p], name_entity[r]))
				size += 1
	print('Accuracy of test dataset: {:.3f}%'.format(n_correct / size * 100.0))
	with open('log.txt', 'w') as fp:
		for p, r in log:
			fp.write('{} | {}\n'.format(p, r))

if __name__ == '__main__':
	args = get_config()
	if args['mode'] == 'train':
		train(args)
	if args['mode'] == 'test':
		test(args)
