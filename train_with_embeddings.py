import tensorflow as tf
import numpy as np
np.set_printoptions(threshold='nan')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import warnings
warnings.simplefilter('error', UserWarning)
from random import shuffle
from random import choice
import json
from PIL import Image
from scipy.misc import imread, imresize
from scipy.stats import truncnorm
import commands
import time
import os
import json

from ops import *
from model import *

import config
import utils

image_metadata = []
same_label_images = {}
def get_image_metadata():
	with open('sanitized_emoji_images_high_quality_medium.json') as data_file:    
		data = json.load(data_file)
		for d in data:
			if d['title'] not in same_label_images:
				same_label_images[d['title']] = [d['filename']]
			else:
				same_label_images[d['title']].append(d['filename'])
		shuffle(data)
		return data

def get_word_vectors():
	with open('word_vectors.json') as data_file:
		return json.load(data_file)

image_metadata = get_image_metadata() # Load all image filenames into memory
word_vectors = get_word_vectors() # Load all word vectors into memory

def get_pixels_for_filename(filename):
	img = imread(filename, mode='RGBA')
	img = imresize(img, [128, 128])
	return np.array(img)

curr_image_idx = 0
def get_next_image_batch(batch_size, same_labels=False):
	global curr_image_idx
	global image_metadata
	global word_vectors

	pixels_batch = np.zeros(shape=(batch_size, config.IMAGE_SIZE))
	embeddings_batch = np.zeros(shape=(batch_size, config.WORD_EMBEDDING_DIM))
	labels = []

	rand_label = choice(same_label_images.keys())
	curr_same_label_index = 0
	for i in range(batch_size):

		if same_labels:
			if curr_same_label_index >= len(same_label_images[rand_label]):
				curr_same_label_index = 0
			pix = get_pixels_for_filename(same_label_images[rand_label][curr_same_label_index])
			labels.append(rand_label)
			pixels_batch[i] = pix.reshape([config.IMAGE_SIZE])
			embeddings_batch[i] = np.array(word_vectors[rand_label])
			curr_same_label_index += 1
		else:
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
				except:
					curr_image_idx += 1
					continue
				if pix.shape != (128, 128, 4):
					print('Invalid pixels shape for file ' + image_metadata[curr_image_idx]['filename'] + ': ' + str(pix.shape) + ". Skipping.")
					# Skip this image
					curr_image_idx += 1
					continue

				label = image_metadata[curr_image_idx]['title']
				if label not in word_vectors:
					print("Didn't find " + label + " in word vectors. Skipping.")
					curr_image_idx += 1
					continue
				pixels_batch[i] = pix.reshape([config.IMAGE_SIZE])
				embeddings_batch[i] = np.array(word_vectors[label])
				labels.append(label)
				curr_image_idx += 1
				break
	return pixels_batch, embeddings_batch, labels

X = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, config.IMAGE_SIZE], name="real_images_input")
X_real_wrong = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, config.IMAGE_SIZE], name="real_images_input_wrong")
z = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 100], name="generator_latent_space_input")
embeddings = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, config.WORD_EMBEDDING_DIM], name="embeddings_input")
instance_noise_std = tf.placeholder(tf.float32, shape=(), name="instance_noise_std")

G = GeneratorWithEmbeddings(z, embeddings, False, 'Generator')
D_real_prob, D_real_logits, feature_matching_real, minibatch_similarity_real = DiscriminatorWithEmbeddings(X, embeddings, instance_noise_std, False, 'Discriminator')
D_real_emoji_wrong_label_prob, D_real_emoji_wrong_label_logits, _, _ = DiscriminatorWithEmbeddings(X_real_wrong, embeddings, instance_noise_std, True, 'Discriminator')
D_fake_prob, D_fake_logits, feature_matching_fake, minibatch_similarity_fake = DiscriminatorWithEmbeddings(G, embeddings, instance_noise_std, True, 'Discriminator')
predicted_z = ModeEncoderWithEmbeddings(X, embeddings, 'ModeEncoder')
image_from_predicted_z = GeneratorWithEmbeddings(predicted_z, embeddings, True, 'Generator')
l2_distance_encoder = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(X, image_from_predicted_z))))
D_mode_regularizer_prob,_,_,_ = DiscriminatorWithEmbeddings(image_from_predicted_z, embeddings, instance_noise_std, True, 'Discriminator')
mode_regularizer_loss = tf.reduce_mean(tf.log(D_mode_regularizer_prob))

D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits) * 0.8), name="disc_real_cross_entropy")
D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)), name="disc_fake_cross_entropy")
D_real_wrong = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_emoji_wrong_label_logits, labels=tf.zeros_like(D_real_emoji_wrong_label_logits)), name="disc_fake_cross_entropy")
D_fake_wrong = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)), name="generator_wrong_fake_cross_entropy")

# This is divided by 16*16 because that's dimension of the intermediate feature we pull out of D
feature_matching_loss = tf.divide(tf.reduce_mean(tf.nn.l2_loss(feature_matching_real - feature_matching_fake)), (float(16) * 16), name="feature_matching_loss")

# WGAN-GP gradient penalty
lambd = 10
alpha = tf.random_uniform(
	shape=[config.BATCH_SIZE,1], 
	minval=0.,
	maxval=1.
)
X_hat = alpha*X + (1-alpha)*G
D_hat,_,_,_ = DiscriminatorWithEmbeddings(X_hat, embeddings, instance_noise_std, True, 'Discriminator')
gradients = tf.gradients(D_hat, [X_hat])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2) * lambd

D_loss = tf.add(D_real/2, D_fake/2, "disc_loss")
D_loss += D_real_wrong

# Commented out: Using minibatch similarity in the loss function. Too difficult to decide exactly how to weight this.
# Minibatch_similarity_loss = tf.nn.l2_loss(minibatch_features_real, minibatch_features_fake, "minibatch_similarity_loss")


# Generator tries to maximize log(D_fake), and I incorporate Feature Matching into the loss function.
# Both techniques are dicussed here: https://arxiv.org/abs/1606.03498
# 0.1 was decided upon via trial/error 
# 
# Actually, for now, no feature matching. Was quite difficult maintaining stability when combining two things in the loss function
# Generator tries to maximize log(D_fake), and I incorporate Feature Matching into the loss function. 
# Also mode regularizer and l2_distance_encoder from MRGAN
# Both techniques are dicussed here: https://arxiv.org/abs/1606.03498
encoder_lambda_1 = 0.01
encoder_lambda_2 = 0.02
feature_matching_lambda = 0.02
l2_distance_encoder *= encoder_lambda_1
mode_regularizer_loss *= encoder_lambda_2
feature_matching_loss *= feature_matching_lambda
G_loss = D_fake_wrong + feature_matching_lambda
E_loss = l2_distance_encoder + mode_regularizer_loss

tf.summary.scalar("D_real_loss", D_real)
tf.summary.scalar("D_fake_loss", D_fake)
tf.summary.scalar("D_fake_wrong", D_fake_wrong)
tf.summary.scalar("gradient_penalty", gradient_penalty)
tf.summary.scalar("feature_matching_loss", feature_matching_loss)
tf.summary.scalar("G_loss", G_loss)
tf.summary.scalar("E_loss", E_loss)
tf.summary.scalar("mode_regularizer_loss", mode_regularizer_loss)
tf.summary.scalar("l2_distance_encoder", l2_distance_encoder)
tf.summary.scalar("instance_noise_std", instance_noise_std)
tf.summary.scalar("minibatch_similarity_real", tf.reduce_mean(minibatch_similarity_real))
tf.summary.scalar("minibatch_similarity_fake", tf.reduce_mean(minibatch_similarity_fake))
tf.summary.scalar("d_real_prob", tf.reduce_mean(D_real_prob))
tf.summary.scalar("d_fake_prob", tf.reduce_mean(D_fake_prob))
tf.summary.scalar("d_real_wrong_prob", tf.reduce_mean(D_real_emoji_wrong_label_prob))

vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('Discriminator/')]
g_params = [v for v in vars if v.name.startswith('Generator/')]
e_params = [v for v in vars if v.name.startswith('ModeEncoder/')]

def train(loss_tensor, params, learning_rate, beta1):
	optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
	grads = optimizer.compute_gradients(loss_tensor, var_list=params)
	for grad, var in grads:
		if grad is not None:
			tf.summary.scalar(var.op.name + "/gradient", tf.reduce_mean(grad))
	return optimizer.apply_gradients(grads)

# Learning rates decided upon by trial/error
disc_optimizer = train(D_loss, d_params, learning_rate=1e-4, beta1=0.5)
generator_optimizer = train(G_loss, g_params, learning_rate=1e-4, beta1=0.5)
# encoder_optimizer = train(E_loss, e_params, learning_rate=1e-4, beta1=0.5)

# Normal distribution centered around 0.0 with stddev 0.33, clipped at -1 and 1
latent_space_sampler = truncnorm(a=-1/0.33, b=1/0.33, scale=0.33)
def get_instance_noise_std(iters_run):
	# Instance noise, motivated by: http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
	# Heuristic: Values are probably best determined by seeing how identifiable
	# your images are with certain levels of noise. Here, I am starting off
	# with INITIAL_NOISE_STD and decreasing uniformly, hitting zero at a threshold iteration.
	INITIAL_NOISE_STD = 0.8
	LAST_ITER_WITH_NOISE = 5000
	if iters_run >= LAST_ITER_WITH_NOISE:
		return 0.0
	return INITIAL_NOISE_STD - ((INITIAL_NOISE_STD/LAST_ITER_WITH_NOISE) * iters_run)

with tf.Session() as sess:
	saver = tf.train.Saver()
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter('tensorboard/',
										 sess.graph)
	sess.run(tf.global_variables_initializer())

	# Try to load model
	curr_step = 0
	could_load, checkpoint_counter = utils.load(config.CHECKPOINT_DIR, sess, saver)
	if could_load:
		curr_step = checkpoint_counter
		print(" [*] Load SUCCESS")
	else:
		print(" [!] Load failed...")

	for e in range(config.EPOCHS):
		for _ in range(config.ITERATIONS):

			instance_noise_std_value = get_instance_noise_std(curr_step)

			x, label_embeddings, labels = get_next_image_batch(config.BATCH_SIZE)
			x_real_wrong, _, _ = get_next_image_batch(config.BATCH_SIZE)
			x_g, label_embeddings_g, labels_g = get_next_image_batch(config.BATCH_SIZE, same_labels=True)
			x = utils.normalize_image_batch(x)
			x_g = utils.normalize_image_batch(x_g)

			rand = latent_space_sampler.rvs((config.BATCH_SIZE, config.Z_DIM))
			feed_dict = {X: x, z: rand, instance_noise_std: instance_noise_std_value, embeddings: label_embeddings, X_real_wrong: x_real_wrong}
			feed_dict_g = {X: x_g, z: rand, instance_noise_std: instance_noise_std_value, embeddings: label_embeddings_g, X_real_wrong: x_real_wrong}
			_, D_loss_curr = sess.run([disc_optimizer, D_loss], feed_dict)

			# Train on same label batch
			# sess.run([generator_optimizer, encoder_optimizer], feed_dict_g)
			# sess.run([disc_optimizer], feed_dict_g)

			# Change latent space for next G
			rand = latent_space_sampler.rvs((config.BATCH_SIZE, config.Z_DIM))
			feed_dict[z] = rand
			if curr_step > 0 and curr_step % config.STEPS_PER_SUMMARY == 0:
				summary, _, G_loss_curr = sess.run([merged, generator_optimizer, G_loss], feed_dict)
				train_writer.add_summary(summary, curr_step)
			else:
				_, G_loss_curr = sess.run([generator_optimizer, G_loss], feed_dict)

			if G_loss_curr > D_loss_curr:
				sess.run([generator_optimizer], feed_dict)
			else:
				sess.run([disc_optimizer], feed_dict)

			sys.stdout.write("\rstep %d: %f, %f" % (curr_step, D_loss_curr, G_loss_curr))
			sys.stdout.flush()
			curr_step += 1

			if curr_step > 0 and curr_step % config.STEPS_PER_IMAGE_SAMPLE == 0:
				# Note that these samples have "pixels" in the range (-1,1)
				generated_samples = sess.run(G, {z: rand, embeddings: label_embeddings})
				utils.save_samples_labeled(generated_samples, labels, curr_step / config.STEPS_PER_IMAGE_SAMPLE)

				# This is for same labels
				generated_samples = sess.run(G, {z: rand, embeddings: label_embeddings_g})
				utils.save_samples_labeled(generated_samples, labels_g, curr_step / config.STEPS_PER_IMAGE_SAMPLE + 1000)

			if curr_step > 0 and curr_step % config.STEPS_PER_SAVE == 0:
				utils.save(config.CHECKPOINT_DIR, curr_step, sess, saver)
