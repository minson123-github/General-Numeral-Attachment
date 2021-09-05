import torch
import pytorch_lightning as pl
from transformers import T5TokenizerFast, T5ForConditionalGeneration

class Seq2SeqNet(pl.LightningModule):
	
	def __init__(self, pretrained_path, lr):
		super().__init__()
		self.model = T5ForConditionalGeneration.from_pretrained(pretrained_path)
		self.tokenizer = T5TokenizerFast.from_pretrained(pretrained_path, bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
		self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
		self.lr = lr
		self.save_hyperparameters()
	
	def training_step(self, batch, batch_idx):
		inputs, labels = batch
		outputs = self.model(input_ids = inputs, labels=labels)
		loss = outputs.loss
		return loss
	
	def validation_step(self, batch, batch_idx):
		inputs, labels = batch
		pred_tokens = self.model.generate(inputs)
		pred = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
		real = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
		n_correct = 0
		for p, r in zip(pred, real):
			if p == r:
				n_correct += 1
		acc = n_correct / len(pred)
		self.log('eval_acc', acc, prog_bar=True)
	
	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
		return optimizer
