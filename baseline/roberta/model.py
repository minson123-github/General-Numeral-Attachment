import torch
import pytorch_lightning as pl
from transformers import RobertaForSequenceClassification

class RobertaClassifier(pl.LightningModule):
	
	def __init__(self, pretrained_path, n_label, lr):
		super().__init__()
		self.model = RobertaForSequenceClassification.from_pretrained(pretrained_path, num_labels=n_label)
		self.lr = lr
		self.save_hyperparameters()
	
	def training_step(self, batch, batch_idx):
		inputs, labels = batch
		outputs = self.model(input_ids = inputs, labels=labels)
		loss = outputs.loss
		pred = []
		for logit in outputs.logits:
			max_p = 0
			for i in range(len(logit)):
				if logit[i] > logit[max_p]:
					max_p = i
			pred.append(max_p)
		real = [label for label in labels]
		n_correct = 0
		for p, r in zip(pred, real):
			if p == r:
				n_correct += 1
		acc = n_correct / len(pred)
		self.log('train_acc', acc, prog_bar=True)
		return loss
	
	def validation_step(self, batch, batch_idx):
		inputs, labels = batch
		outputs = self.model(input_ids = inputs, labels=labels)
		loss = outputs.loss
		pred = []
		for logit in outputs.logits:
			max_p = 0
			for i in range(len(logit)):
				if logit[i] > logit[max_p]:
					max_p = i
			pred.append(max_p)
		real = [label for label in labels]
		n_correct = 0
		for p, r in zip(pred, real):
			if p == r:
				n_correct += 1
		acc = n_correct / len(pred)
		self.log('eval_acc', acc, prog_bar=True)
		return loss
	
	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
		return optimizer
