import numpy as np
import tensorflow as tf
import commands
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from random import shuffle
import json
import config
import scipy.misc

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

image_metadata = []

def get_image_metadata():
	with open('emoji_images_high_quality_medium.json') as data_file:    
		data = json.load(data_file)
		shuffle(data)
		return data

image_metadata = get_image_metadata() # Load all image filenames into memory

def get_pixels_for_filename(filename):
    img = scipy.misc.imread(filename, mode='RGBA')
    img = scipy.misc.imresize(img, [64, 64])
    return np.array(img)

curr_image_idx = 0
def get_next_image_batch(batch_size):
	global curr_image_idx
	global image_metadata

	pixels_batch = np.zeros(shape=(batch_size, config.IMAGE_SIZE))
	for i in range(batch_size):

		# Keep looping til we get an image. Sometimes we'll find an image with an invalid size.
		# TODO: Just sanitize the data
		while True:

			# Go back to the start, and then shuffle the images
			if curr_image_idx >= len(image_metadata):
				curr_image_idx = 0
				shuffle(image_metadata)
			# Note that we don't store all pixels in memory bec of memory constraints

			try: # Another hack bec the metadata has stale files that have been deleted
				pix = get_pixels_for_filename(image_metadata[curr_image_idx]['filename'])
			except Exception as e:
				print(str(e))
				curr_image_idx += 1
				continue
			if pix.shape != (64, 64, 4):
				print('Invalid pixels shape for file ' + image_metadata[curr_image_idx]['filename'] + ': ' + str(pix.shape))
				# Skip this image
				curr_image_idx += 1
				continue

			pixels_batch[i] = pix.reshape([config.IMAGE_SIZE])
			curr_image_idx += 1
			break
	return pixels_batch

def save_samples(samples, image_num, is_test=False):
	fig = plt.figure(figsize=(18, 18))
	gs = gridspec.GridSpec(8, 8)
	gs.update(wspace=0.06, hspace=0.06)

	# Generate images
	for i, sample in enumerate(samples[:64]):
		# need to convert sample from range -1,1 to 0 255
		sample = denormalize_image(sample)
		ax = plt.subplot(gs[i % 8, int(i / 8)])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(64, 64, 4).astype(np.uint8))

	if is_test:
		plt.savefig('./test.png', bbox_inches='tight')
		print("New sample: ./test.png")
	else:
		plt.savefig('./output/' + str(image_num) + '.png', bbox_inches='tight')
		print('New samples: ./output/' + str(image_num) + '.png')
	plt.close()

def save(checkpoint_dir, curr_step, sess, saver):
	print(" [*] Saving model at step: " + str(curr_step))
	checkpoint_dir = os.path.join(checkpoint_dir, config.MODEL_DIR)

	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	saver.save(sess,
				os.path.join(checkpoint_dir, config.MODEL_NAME),
				global_step=curr_step)
	print(" [*] Successfully saved model")

def load(checkpoint_dir, sess, saver):
	import re
	print(" [*] Reading checkpoints...")
	checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, config.MODEL_DIR)

	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
		curr_step = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
		print(" [*] Successfully read {}".format(ckpt_name))
		return True, curr_step
	else:
		print(" [*] Failed to find a checkpoint")
	return False, 0