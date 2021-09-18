import os
import random
import numpy as np
import tensorflow as tf
from config import get_config
from transformers import RobertaTokenizerFast, AdamWeightDecay
from utils.model import RoBERTaNumeralClassifier, AccumulateAccuracyMetric, SaveCallback
from utils.data_process import get_all_category, get_train_dataset, get_eval_dataset, get_test_dataset

tf.get_logger().setLevel('ERROR')
tf.compat.v1.enable_eager_execution()
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices:
	tf.config.experimental.set_memory_growth(gpu_instance, True)

def fix_seed(seed):
	print('Setting global seed to {}'.format(seed))
	np.random.seed(seed)
	random.seed(seed)
	tf.random.set_seed(seed)

def train(args):
	fix_seed(args['seed'])
	category = get_all_category(args)
	tokenizer = RobertaTokenizerFast.from_pretrained(args['pretrained_model'])
	train_dataset = get_train_dataset(args, tokenizer)
	model = RoBERTaNumeralClassifier(args['pretrained_model'], len(category))
#	model.load_weights('ckpt/classifier')
	model.compile(
		optimizer=AdamWeightDecay(learning_rate=args['lr']), 
		loss=tf.keras.losses.SparseCategoricalCrossentropy(
			from_logits=True, reduction=tf.keras.losses.Reduction.NONE
		), 
		metrics=[AccumulateAccuracyMetric()], 
		run_eagerly=True
	)
	save_dir = os.path.join(args['model_dir'], 'model')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	ckpt_dir = os.path.join(args['model_dir'], 'checkpoints')
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	model.fit(
		train_dataset, 
		epochs=args['n_epoch'], 
		verbose=1, 
		workers=4, 
		callbacks=SaveCallback(args['saving_steps'], args['model_dir'])
	)
	save_path = os.path.join(save_dir, 'classifier')
	model.save_weights(save_path)

def test(args):
	fix_seed(args['seed'])
	category = get_all_category(args)
	tokenizer = RobertaTokenizerFast.from_pretrained(args['pretrained_model'])
	test_dataset = get_test_dataset(args, tokenizer)
	eval_dataset = get_eval_dataset(args, tokenizer)
	model = RoBERTaNumeralClassifier(args['pretrained_model'], len(category))
	model.load_weights(args['testing_model'])
	model.compile(
		metrics=[AccumulateAccuracyMetric()], 
		run_eagerly=True
	)

	model.evaluate(test_dataset, verbose=1, workers=4)
	model.evaluate(eval_dataset, verbose=1, workers=4)

if __name__ == '__main__':
	args = get_config()
	if args['mode'] == 'train':
		train(args)
	if args['mode'] == 'test':
		test(args)
