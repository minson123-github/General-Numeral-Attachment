import torch
from transformers import RobertaModel, RobertaPreTrainedModel

class RobertaNumeralClassifier(RobertaPreTrainedModel):
	_keys_to_ignore_on_load_unexpected = [r"pooler"]
	_keys_to_ignore_on_load_missing = [r"position_ids"]

	def __init__(self, config):
		super().__init__(config)
		self.roberta = RobertaModel(config, add_pooling_layer=False)
		classifier_dropout = config.hidden_dropout_prob
		self.dropout = torch.nn.Dropout(classifier_dropout)
		self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
		self.init_weights()
	
	def forward(self, x, numeral_position):
		outputs = self.roberta(x)
		hidden_states = outputs[0]
		x = torch.stack([hidden_states[idx, p, :] for idx, p in enumerate(numeral_position)])
		x = self.dropout(x)
		x = self.classifier(x)
		return x
