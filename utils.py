import numpy as np
import tensorflow as tf

# Convert images in batch from having values [0,255] to (-1,1)
def normalize_image_batch(image_batch):
	normalized_batches = np.zeros(image_batch.shape)
	for idx, batch in enumerate(image_batch):
		normalized_batches[idx] = np.multiply(2, np.divide(batch, float(255))) - 1
	return normalized_batches

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	tf.summary.histogram(var.op.name, var)

def denormalize_image(image):
	return np.multiply(np.divide((1 + image), 2), 255)

# Convert images in batch from having values [0,255] to (-1,1)
def normalize_image_batch(image_batch):
	normalized_batches = np.zeros(image_batch.shape)
	for idx, batch in enumerate(image_batch):
		normalized_batches[idx] = np.multiply(2, np.divide(batch, float(255))) - 1
	return normalized_batches