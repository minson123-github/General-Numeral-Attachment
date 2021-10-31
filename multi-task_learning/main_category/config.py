import argparse

def get_config():
	parser = argparse.ArgumentParser()
	### data process config
	parser.add_argument('--data', type=str, help='File contains all data.')
	parser.add_argument('--train_data', type=str, help='File contains training data.')
	parser.add_argument('--eval_data', type=str, help='File contains evaluate data.')
	parser.add_argument('--test_data', type=str, help='File contains testing data.')
	### ml config
	parser.add_argument('--mode', type=str, help='Mode: train or test')
	parser.add_argument('--pretrained_model', type=str, default='roberta-large', help='Pretrained RoBERTa model.')
	parser.add_argument('--testing_model', type=str, help='Model for inference.')
	parser.add_argument('--padding_length', type=int, default=512, help='tokenize padding length.')
	parser.add_argument('--extend_token', type=int, help='Number of extra token before target numeral and after target numeral.')
	parser.add_argument('--batch_size', type=int, help='Batch size in dataloader parameter.')
	parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of step for gradient accumulate.')
	parser.add_argument('--lr', type=float, help='Learning rate for optimizer.')
	parser.add_argument('--n_epoch', type=int, default=1, help='Number of epoch for training.')
	parser.add_argument('--seed', type=int, default=24, help='random seed.')
	parser.add_argument('--model_dir', type=str, default='ckpt', help='The directory for saving model and checkpoints.')
	parser.add_argument('--saving_steps', type=int, help='Number of steps to save checkpoint.')
	parser.add_argument('--saving_epochs', type=int, help='Number of epochs to save checkpoint.')
	args = parser.parse_args()
	return vars(args)
