import torch
from transformers import RobertaModel, RobertaPreTrainedModel

class RobertaNumeralQA(RobertaPreTrainedModel):
	_keys_to_ignore_on_load_unexpected = [r"pooler"]
	_keys_to_ignore_on_load_missing = [r"position_ids"]

	def __init__(self, config):
		super().__init__(config)
		self.roberta = RobertaModel(config, add_pooling_layer=False)
		classifier_dropout = config.hidden_dropout_prob
		self.dropout = torch.nn.Dropout(classifier_dropout)
		self.qa_classifier = torch.nn.Linear(config.hidden_size * 2, config.num_labels)
		self.init_weights()
	
	def forward(self, x, numeral_position):
		outputs = self.roberta(x)
		hidden_states = outputs[0]
		mask_hidden_state = torch.stack([hidden_states[idx, p, :] for idx, p in enumerate(numeral_position)])
		x = torch.cat(torch.broadcast_tensors(hidden_states, mask_hidden_state.unsqueeze(dim=1)), dim=-1)
		x = self.dropout(x)
		logits = self.qa_classifier(x)
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1).contiguous()
		end_logits = end_logits.squeeze(-1).contiguous()
		return start_logits, end_logits
