import tensorflow as tf
import numpy as np
np.set_printoptions(threshold='nan')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import warnings
warnings.simplefilter('error', UserWarning)
from random import shuffle
import json
from PIL import Image
import scipy.misc
import commands
import time

ITERATIONS = 10
BATCH_SIZE = 64
EPOCHS = 100000
IMAGE_SIZE = 32*32*4

image_filenames = []

def get_image_filenames():
	s = commands.getstatusoutput('ls medium_small_data')
	filenames = ['medium_small_data/' + string for string in s[1].split()]
	shuffle(filenames)
	return filenames
image_filenames = get_image_filenames() # Load all image filenames into memory

def get_pixels_for_filename(filename):
    img = scipy.misc.imread(filename, mode='RGBA')
    return np.array(img)

curr_image_idx = 0
def get_next_image_batch(batch_size):
	global curr_image_idx
	global image_filenames

	batch = np.zeros(shape=(batch_size, IMAGE_SIZE))
	for i in range(batch_size):

		# Keep looping til we get an image. Sometimes we'll find an image with an invalid size.
		# TODO: Just sanitize the data
		while True:

			# Go back to the start, and then shuffle the images
			if curr_image_idx >= len(image_filenames):
				curr_image_idx = 0
				shuffle(image_filenames)
			# Note that we don't store all pixels in memory bec of memory constraints
			pix = get_pixels_for_filename(image_filenames[curr_image_idx])
			if pix.shape != (32, 32, 4):
				print('Invalid pixels shape for file ' + image_filenames[curr_image_idx] + ': ' + str(pix.shape))
				# Skip this image
				curr_image_idx += 1
				continue

			batch[i] = pix.reshape([IMAGE_SIZE])
			curr_image_idx += 1
			break
	return batch

def normalize_image_batches(image_batches):
	normalized_batches = np.zeros(image_batches.shape)
	for idx, batch in enumerate(image_batches):
		normalized_batches[idx] = np.multiply(2, np.divide(batch, float(255))) - 1
	return normalized_batches

def add_gaussian_noise(normalized_image):
	return np.clip(normalized_image + np.random.normal(0, 0.22, normalized_image.size), a_min=-1, a_max=1)

def denormalize_image(image):
	return np.multiply(np.divide((1 + image), 2), 255)

def plot(samples, D_loss, G_loss, epoch, total):
	fig = plt.figure(figsize=(18, 18))
	# First 4 columns are images, last 4 is for the loss
	gs = gridspec.GridSpec(8, 8)
	gs.update(wspace=0.06, hspace=0.06)

	# Generate images
	for i, sample in enumerate(samples):
		# need to convert sample from range -1,1 to 0 255
		sample = denormalize_image(sample)
		# Plot in the left half
		ax = plt.subplot(gs[i % 8, int(i / 8)])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(32, 32, 4).astype(np.uint8))
	plt.savefig('./test.png', bbox_inches='tight')

batch = get_next_image_batch(64)
batch = normalize_image_batches(batch)
for i, nb in enumerate(batch):
	batch[i] = add_gaussian_noise(nb)

plot(batch, [1, 1, 1], [1, 1, 1], 1, 1)