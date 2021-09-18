import os
import tensorflow as tf
from transformers import TFRobertaModel
from transformers.modeling_tf_utils import get_initializer

class RoBERTaNumeralClassifier(tf.keras.Model):
	_keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
	
	def __init__(self, pretrained_path, num_labels):
		super(RoBERTaNumeralClassifier, self).__init__()
		self.roberta = TFRobertaModel.from_pretrained(pretrained_path)
		dropout_prob = self.roberta.config.hidden_dropout_prob
		self.dropout = tf.keras.layers.Dropout(dropout_prob)
		self.classifier = tf.keras.layers.Dense(
			num_labels, kernel_initializer=get_initializer(self.roberta.config.initializer_range), name="classifier"
		)
	
	def call(self, batch, training=False):
		x = batch['paragraph']
		numeral_position = batch['numeral_position']
		outputs = self.roberta(x, training=training)
		x = [tf.reshape(tf.slice(outputs[0], [idx, p[0], 0], [1, 1, -1]), [-1]) for idx, p in enumerate(numeral_position.numpy())]
		x = self.dropout(x, training=training)
		x = self.classifier(x)
		return x

class AccumulateAccuracyMetric(tf.keras.metrics.Metric):
	
	def __init__(self, name='accumulate_accuracy_metric', **kwargs):
		super(AccumulateAccuracyMetric, self).__init__(name=name, **kwargs)
		self.n_correct = self.add_weight(name='nc', initializer='zeros', dtype='int32')
		self.n_sample = self.add_weight(name='ns', initializer='zeros', dtype='int32')
	
	def update_state(self, y_true, y_pred, sample_weight=None):
		y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
		values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
		values = tf.cast(values, 'int32')
		self.n_correct.assign_add(tf.reduce_sum(values))
		self.n_sample.assign_add(len(values))
	
	def result(self):
		return tf.divide(tf.cast(self.n_correct, dtype='float32'), tf.cast(self.n_sample, dtype='float32'))
	
	def reset_state(self):
		self.n_correct.assign(0)
		self.n_sample.assign(0)

class SaveCallback(tf.keras.callbacks.Callback):
	
	def __init__(self, save_freq, model_dir):
		super(SaveCallback, self).__init__()
		self.n_step = 0
		self.save_freq = save_freq
		self.model_dir = model_dir

	def on_batch_end(self, batch, logs=None):
		self.n_step += 1
		if self.n_step % self.save_freq == 0:
			ckpt_path = os.path.join(self.model_dir, 'checkpoints', 'ckpt_{}'.format(self.n_step // self.save_freq))
			self.model.save_weights(ckpt_path)
