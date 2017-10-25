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
import os

from ops import *
import config

BATCH_SIZE = 64

def save_samples(samples):
	fig = plt.figure(figsize=(18, 18))
	gs = gridspec.GridSpec(8, 8)
	gs.update(wspace=0.06, hspace=0.06)

	# Generate images
	for i, sample in enumerate(samples):
		# need to convert sample from range -1,1 to 0 255
		sample = denormalize_image(sample)
		ax = plt.subplot(gs[i % 8, int(i / 8)])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(32, 32, 4).astype(np.uint8))
	
	plt.savefig('./sample_output.png', bbox_inches='tight')
	print('New samples: ./sample_output.png')
	plt.close()

def Generator(z, name='g'):
	# Architecture:
	# 	Project to 1024*4*4 then reshape, then BN
	# 	Then deconv with stride 2, 5x5 filters into 512*8*8, then BN
	# 	Then deconv with stride 2, 5x5 filters into 256*16*16, then BN
	# 	Then deconv with stride 2, 5x5 filters into 4*32*32
	#   tanh
	with tf.variable_scope(name):

		G_1 = Dense(z, output_dim=1024*4*4, name='dense')
		G_r1 = tf.reshape(G_1, [BATCH_SIZE, 4, 4, 1024])
		G_bn1 = BatchNormalization(G_r1, name='dense_bn')
		G_h1 = tf.nn.relu(G_bn1)
		with tf.name_scope('dense_activation'):
			variable_summaries(G_h1)

		G_conv2 = Deconv2d(G_h1, output_dim=512, batch_size=BATCH_SIZE, name='deconv1')
		G_bn2 = BatchNormalization(G_conv2, name='deconv1_bn')
		G_h2 = tf.nn.relu(G_bn2)
		with tf.name_scope('deconv1_activation'):
			variable_summaries(G_h2)

		G_conv3 = Deconv2d(G_h2, output_dim=256, batch_size=BATCH_SIZE, name='deconv2')
		G_bn3 = BatchNormalization(G_conv3, name='deconv2_bn')
		G_h3 = tf.nn.relu(G_bn3)
		with tf.name_scope('deconv2_activation'):
			variable_summaries(G_h3)

		G_conv4 = Deconv2d(G_h3, output_dim=4, batch_size=BATCH_SIZE, name='deconv3')
		G_r4 = tf.reshape(G_conv4, [BATCH_SIZE, 32*32*4])
		tanh_layer = tf.nn.tanh(G_r4)
		with tf.name_scope('tanh'):
			variable_summaries(tanh_layer)
		return tanh_layer

z = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 100], name="generator_latent_space_input")
G = Generator(z, 'Generator')

saver = tf.train.Saver()
def load(checkpoint_dir, sess):
	global saver
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

def generate_random_walk_vector():
	ret = np.zeros((config.BATCH_SIZE, 100))
	curr = np.random.rand(100)
	for i in range(config.BATCH_SIZE):
		ret[i] = curr
		curr = np.clip(curr + np.random.rand(100)/750, a_min=0, a_max=1)
		print(curr)
	return ret

def generate_random_walk_one_dimension():
	ret = np.zeros((config.BATCH_SIZE, 100))
	curr = np.random.rand(100)
	rand_dimension = np.random.randint(100)
	should_add = curr[rand_dimension] < 0.5
	for i in range(config.BATCH_SIZE):
		ret[i] = curr
		if should_add:
			curr += (0.05 / (config.BATCH_SIZE + 1))
		else:
			curr -= (0.05 / (config.BATCH_SIZE + 1))
	return ret

def generate_random_walk_arithmetic():
	ret = np.zeros((config.BATCH_SIZE, 100))
	curr = np.random.rand(100)
	rand_dimension = np.random.randint(100)
	should_add = curr[rand_dimension] < 0.5

	first = curr
	for i in range(config.BATCH_SIZE-8):
		ret[i] = curr
		if should_add:
			curr += (0.1 / (config.BATCH_SIZE + 1))
		else:
			curr -= (0.1 / (config.BATCH_SIZE + 1))

	diff = first - curr
	for i in range(config.BATCH_SIZE-8, config.BATCH_SIZE):
		ret[i] = np.random.rand() + diff

	return ret

with tf.Session() as sess:
	could_load, checkpoint_counter = load(config.CHECKPOINT_DIR, sess)
	if could_load:
		curr_step = checkpoint_counter
		print(" [*] Load success. Attempting to generate samples")
	else:
		print(" [!] Load failed...")
		sys.exit()

	rand_init = np.random.rand()
	# rand = np.random.uniform(rand_init, rand_init+0.1, size=[config.BATCH_SIZE, 100]).astype(np.float32)
	generated_samples = sess.run(G, {z: generate_random_walk_one_dimension()})
	save_samples(generated_samples)

