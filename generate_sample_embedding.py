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
from gensim.models import KeyedVectors
from ops import *
from model import *
import re
import utils
import config
from scipy.stats import truncnorm

z = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 100], name="generator_latent_space_input")
embeddings = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, config.WORD_EMBEDDING_DIM], name="embeddings_input")
G = GeneratorWithEmbeddings(z, embeddings, False, 'Generator')

def generate_random_label():
	with open('sanitized_emoji_images_high_quality_medium.json') as data_file:    
		e = json.load(data_file)
		with open('word_vectors.json') as wv:
			v = json.load(wv)
	random_pair = e[np.random.randint(len(e))]
	vector = v[random_pair['title']]
	return random_pair['title'], vector

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

has_loaded_word2vec = False
model = None
def generate_with_query():
	global has_loaded_word2vec
	global model
	if not has_loaded_word2vec:
		print(" [*] Loading word2vec model...")
		model = KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
		has_loaded_word2vec = True
	print("Enter query:")
	q = sys.stdin.readline().strip()
	embedding = try_to_get_embedding(q, model)
	return q, embedding

def try_to_get_embedding(query, model):
	print("Searching for embedding for: " + query)
	if query in model:
		print("Found embedding for: " + query)
		return model[query]
	else:
		cc = '_'.join(query.split(' '))
		print("Searching for embedding for: " + cc)
		if cc in model:
			print("Found embedding for: " + cc)
			return model[cc]
		else:
			replaced = cc.replace('-', '_')
			print("Searching for embedding for: " + replaced)
			if replaced in model:
				print("Found embedding for: " + replaced)
				return model[replaced]
			else:
				print("Trying to get average of all words in query that exist in word2vec...")
				stripped = re.sub('[^A-Za-z0-9]+', ' ', cc)
				words = stripped.split(' ')
				words_found = 0
				words_in_model = []
				total = np.zeros(300)
				for word in words:
					if word in model:
						words_in_model.append(word)
						words_found += 1
						total = np.add(model[word], total)
				if words_found == 0:
					print("Could not find embedding in word2vec model")
					sys.exit()
				print("Taking average of embeddings for: " + str(words_in_model))
				return np.divide(total, words_found)

with tf.Session() as sess:
	saver = tf.train.Saver()
	could_load, checkpoint_counter = utils.load(config.CHECKPOINT_DIR, sess, saver)
	if could_load:
		curr_step = checkpoint_counter
		print(" [*] Successfully loaded model")
	else:
		print(" [!] Load failed...")
		sys.exit()

	while True:
		print("CTRL+C to quit")
		latent_space_sampler = truncnorm(a=-1/0.33, b=1/0.33, scale=0.33)
		rand = latent_space_sampler.rvs((config.BATCH_SIZE, config.Z_DIM))
		# label, vector = generate_random_label()
		label, vector = generate_with_query()

		embeddings_batch = np.zeros(shape=(config.BATCH_SIZE, config.WORD_EMBEDDING_DIM))
		for i,_ in enumerate(embeddings_batch):
			embeddings_batch[i] = np.array(vector)
		generated_samples = sess.run(G, {z: rand, embeddings: embeddings_batch})
		save_samples(generated_samples, 0, True, label)


