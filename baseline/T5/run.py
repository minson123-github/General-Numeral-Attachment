import os
import torch
from tqdm import tqdm
from model import Seq2SeqNet
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from config import get_config
from pytorch_lightning import Trainer, seed_everything, Callback
from data_process import get_train_dataloader, get_eval_dataloader, get_test_dataloader

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
	tokenizer = T5TokenizerFast.from_pretrained(args['pretrained_model'], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
	model = Seq2SeqNet(args['pretrained_model'], args['lr'])

	train_dataloader = get_train_dataloader(args, tokenizer)
	eval_dataloader = get_eval_dataloader(args, tokenizer)

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
		lr_finder = trainer.tuner.lr_find(model, train_dataloader)
		print(lr_finder.results)
		new_lr = lr_finder.suggestion()
		print('The best learning rate is:', new_lr)
		model.hparams.lr = new_lr
	
	print('The learning rate be used for training:', model.lr, flush=True)
	trainer.fit(model, train_dataloader, eval_dataloader)
	model.model.save_pretrained(save_path)

def test(args):
	seed_everything(args['seed'], workers=True)
	model = Seq2SeqNet.load_from_checkpoint(args['testing_model'])
	model.to('cuda')
	tokenizer = T5TokenizerFast.from_pretrained(args['pretrained_model'], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
	test_dataloader = get_train_dataloader(args, tokenizer)
	n_correct = 0
	size = 0
	log = []
	model.eval()
	with torch.no_grad():
		for (inputs, labels) in tqdm(test_dataloader, position=0, desc='inference'):
			pred_tokens = model.model.generate(inputs.cuda())
			pred = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
			real = tokenizer.batch_decode(labels, skip_special_tokens=True)
			for p, r in zip(pred, real):
				if p == r:
					n_correct += 1
				log.append((p, r))
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
