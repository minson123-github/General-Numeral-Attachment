import argparse

def get_config():
	parser = argparse.ArgumentParser()
	### data process config
	parser.add_argument('--train_data', type=str, help='File contains training data.')
	parser.add_argument('--eval_data', type=str, help='File contains evaluate data.')
	parser.add_argument('--test_data', type=str, help='File contains testing data.')
	### ml config
	parser.add_argument('--mode', type=str, help='Mode: train or test')
	parser.add_argument('--auto_lr', type=int, default=0, help='Whether to find learning rate by pytorch lightning package.')
	parser.add_argument('--pretrained_model', type=str, default='t5-base', help='Pretrained T5 model.')
	parser.add_argument('--testing_model', type=str, help='Model for inference.')
	parser.add_argument('--batch_size', type=int, help='Batch size in dataloader parameter.')
	parser.add_argument('--lr', type=float, help='Learning rate for optimizer.')
	parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPU for training.')
	parser.add_argument('--n_epoch', type=int, default=1, help='Number of epoch for training.')
	parser.add_argument('--seed', type=int, default=24, help='random seed.')
	parser.add_argument('--model_dir', type=str, default='ckpt', help='The directory for saving model and checkpoints.')
	parser.add_argument('--saving_steps', type=int, help='Number of steps to save checkpoint.')
	args = parser.parse_args()
	return vars(args)
