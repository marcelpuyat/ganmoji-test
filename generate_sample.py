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
from model import *

import utils
import config

z = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 100], name="generator_latent_space_input")
G = Generator(z, 'Generator')

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
	saver = tf.train.Saver()
	could_load, checkpoint_counter = utils.load(config.CHECKPOINT_DIR, sess, saver)
	if could_load:
		curr_step = checkpoint_counter
		print(" [*] Load success. Attempting to generate samples")
	else:
		print(" [!] Load failed...")
		sys.exit()

	rand_init = np.random.rand()
	generated_samples = sess.run(G, {z: generate_random_walk_one_dimension()})
	save_samples(generated_samples, 0, True)

