import math
import random
import numpy as np
import tensorflow as tf

def create_dataset(paragraph, numeral_position, category_id, batch_size, shuffle):
	dataset = tf.data.Dataset.from_tensor_slices(({
		'paragraph': paragraph, 
		'numeral_position': numeral_position
	}, category_id)).batch(batch_size)
	if shuffle:
		dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=True)
	
	return dataset
