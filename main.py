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
from utils import *
import config

ITERATIONS = 100
EPOCHS = 1000
STEPS_PER_SUMMARY = 5
STEPS_PER_IMAGE_SAMPLE = 10
STEPS_PER_SAVE = 100

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

	batch = np.zeros(shape=(batch_size, config.IMAGE_SIZE))
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

			batch[i] = pix.reshape([config.IMAGE_SIZE])
			curr_image_idx += 1
			break
	return batch

def save_samples(samples, image_num):
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

	plt.savefig('./output/' + str(image_num) + '.png', bbox_inches='tight')
	print('New samples: ./output/' + str(image_num) + '.png')
	plt.close()

def Discriminator(X, instance_noise_std, reuse=False, name='d'):
	# Architecture:
	# 	Add noise
	# 	Conv3x3, BN, ReLU
	# 	Minibatch discrim computed (sent FC layer at the end)
	# 	Conv3x3, BN, ReLU
	# 	Conv3x3, BN, ReLU
	# 	Conv3x3, BN, ReLU
	# 	FC layer
	# 	Sigmoid
	with tf.variable_scope(name, reuse=reuse):
		# Decaying noise
		X = gaussian_noise_layer(X, instance_noise_std)
		if len(X.get_shape()) > 2:
			# X: -1, 32, 32, 4
			D_conv1 = Conv2d(X, output_dim=32, kernel=(3,3), name='conv1')
		else:
			D_reshaped = tf.reshape(X, [config.BATCH_SIZE, 32, 32, 4])
			D_conv1 = Conv2d(D_reshaped, output_dim=32, kernel=(3,3), name='conv1')

		D_conv1_reshaped = tf.reshape(D_conv1, [config.BATCH_SIZE, -1])
		minibatch_features = minibatch(D_conv1_reshaped) # Saved for the end

		D_bn1 = BatchNormalization(D_conv1, name='conv_bn1')
		D_h1 = LeakyReLU(D_bn1)
		D_conv2 = Conv2d(D_h1, output_dim=64, kernel=(3,3), name='conv2')
		D_bn2 = BatchNormalization(D_conv2, name='conv_bn2')
		D_h2 = LeakyReLU(D_bn2)
		D_conv3 = Conv2d(D_h2, output_dim=128, kernel=(3,3), name='conv3')
		D_bn3 = BatchNormalization(D_conv3, name='conv_bn3')
		D_h3 = LeakyReLU(D_bn3)
		D_conv4 = Conv2d(D_h3, output_dim=256, kernel=(3,3), name='conv4')
		D_bn4 = BatchNormalization(D_conv4, name='conv_bn4')
		D_h4 = LeakyReLU(D_bn4)

		D_r = tf.reshape(D_h4, [config.BATCH_SIZE, 1024])

		# Apply strong dropout on minibatch features because we care less about it compared to image features
		minibatch_features_dropped_out = tf.nn.dropout(minibatch_features, 0.4)

		# Only a bit of dropout for image features to prevent overfitting
		D_r_dropped_out = tf.nn.dropout(D_r, 0.8)

		D_5 = tf.concat([D_r_dropped_out, minibatch_features_dropped_out], 1)

		D_h6 = Dense(D_5, output_dim=1, name='dense')
		preds = tf.nn.sigmoid(D_h6, name='predictions')
		return preds, D_h6, D_h4, minibatch_features

def Generator(z, name='g'):
	# Architecture:
	# 	Project to 1024*4*4 then reshape, then BN
	# 	Then deconv with stride 2, 5x5 filters into 512*8*8, then BN
	# 	Then deconv with stride 2, 5x5 filters into 256*16*16, then BN
	# 	Then deconv with stride 2, 5x5 filters into 4*32*32
	#   tanh
	with tf.variable_scope(name):

		G_1 = Dense(z, output_dim=1024*4*4, name='dense')
		G_r1 = tf.reshape(G_1, [config.BATCH_SIZE, 4, 4, 1024])
		G_bn1 = BatchNormalization(G_r1, name='dense_bn')
		G_h1 = tf.nn.relu(G_bn1)
		with tf.name_scope('dense_activation'):
			variable_summaries(G_h1)

		G_conv2 = Deconv2d(G_h1, output_dim=512, batch_size=config.BATCH_SIZE, name='deconv1')
		G_bn2 = BatchNormalization(G_conv2, name='deconv1_bn')
		G_h2 = tf.nn.relu(G_bn2)
		with tf.name_scope('deconv1_activation'):
			variable_summaries(G_h2)

		G_conv3 = Deconv2d(G_h2, output_dim=256, batch_size=config.BATCH_SIZE, name='deconv2')
		G_bn3 = BatchNormalization(G_conv3, name='deconv2_bn')
		G_h3 = tf.nn.relu(G_bn3)
		with tf.name_scope('deconv2_activation'):
			variable_summaries(G_h3)

		G_conv4 = Deconv2d(G_h3, output_dim=4, batch_size=config.BATCH_SIZE, name='deconv3')
		G_r4 = tf.reshape(G_conv4, [config.BATCH_SIZE, 32*32*4])
		tanh_layer = tf.nn.tanh(G_r4)
		with tf.name_scope('tanh'):
			variable_summaries(tanh_layer)
		return tanh_layer

X = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, config.IMAGE_SIZE], name="real_images_input")
z = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 100], name="generator_latent_space_input")
instance_noise_std = tf.placeholder(tf.float32, shape=(), name="instance_noise_std")

G = Generator(z, 'Generator')
D_real_prob, D_real_logits, feature_matching_real, minibatch_similarity_real = Discriminator(X, instance_noise_std, False, 'Discriminator')
D_fake_prob, D_fake_logits, feature_matching_fake, minibatch_similarity_fake = Discriminator(G, instance_noise_std, True, 'Discriminator')

tf.summary.histogram("d_real_prob", D_real_prob)
tf.summary.histogram("d_fake_prob", D_fake_prob)

D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)), name="disc_real_cross_entropy")
D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)), name="disc_fake_cross_entropy")
D_fake_wrong = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)), name="generator_wrong_fake_cross_entropy")

# This is divided by 16*16 because that's dimension of the intermediate feature we pull out of D
feature_matching_loss = tf.divide(tf.reduce_mean(tf.nn.l2_loss(feature_matching_real - feature_matching_fake)), (float(16) * 16), name="feature_matching_loss")

D_loss = tf.add(D_real, D_fake, "disc_loss")

# Commented out: Using minibatch similarity in the loss function. Too difficult to decide exactly how to weight this.
# Minibatch_similarity_loss = tf.nn.l2_loss(minibatch_features_real, minibatch_features_fake, "minibatch_similarity_loss")


# Generator tries to maximize log(D_fake), and I incorporate Feature Matching into the loss function.
# Both techniques are dicussed here: https://arxiv.org/abs/1606.03498
# 0.1 was decided upon via trial/error 
# 
# Actually, for now, no feature matching. Was quite difficult maintaining stability when combining two things in the loss function
G_loss = tf.identity(D_fake_wrong, "generator_loss")

tf.summary.scalar("D_real_loss", D_real)
tf.summary.scalar("D_fake_loss", D_fake)
tf.summary.scalar("feature_matching_loss", feature_matching_loss)
tf.summary.histogram("minibatch_similarity_real", minibatch_similarity_real)
tf.summary.histogram("minibatch_similarity_fake", minibatch_similarity_fake)
tf.summary.scalar("G_loss", G_loss)
tf.summary.scalar("instance_noise_std", instance_noise_std)

vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('Discriminator/')]
g_params = [v for v in vars if v.name.startswith('Generator/')]

def train(loss_tensor, params, learning_rate, beta1):
	optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
	grads = optimizer.compute_gradients(loss_tensor, var_list=params)
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + "/gradient", grad)
	return optimizer.apply_gradients(grads)

# Learning rates decided upon by trial/error
disc_optimizer = train(D_loss, d_params, learning_rate=5e-4, beta1=0.5)
generator_optimizer = train(G_loss, g_params, learning_rate=1e-3, beta1=0.5)

def get_instance_noise_std(iters_run):
	# Instance noise, motivated by: http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
	# Heuristic: Values are probably best determined by seeing how identifiable
	# your images are with certain levels of noise. Here, I am starting off
	# with INITIAL_NOISE_STD and decreasing uniformly, hitting zero at a threshold iteration.
	INITIAL_NOISE_STD = 0.3
	LAST_ITER_WITH_NOISE = 300
	if iters_run >= LAST_ITER_WITH_NOISE:
		return 0.0
	return INITIAL_NOISE_STD - ((INITIAL_NOISE_STD/LAST_ITER_WITH_NOISE) * iters_run)



saver = tf.train.Saver()
def save(checkpoint_dir, curr_step, sess):
	global saver
	print(" [*] Saving model at step: " + str(curr_step))
	checkpoint_dir = os.path.join(checkpoint_dir, config.MODEL_DIR)

	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	saver.save(sess,
				os.path.join(checkpoint_dir, config.MODEL_NAME),
				global_step=curr_step)
	print(" [*] Successfully saved model")

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

with tf.Session() as sess:
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter('tensorboard/',
										 sess.graph)
	sess.run(tf.global_variables_initializer())

	# Try to load model
	curr_step = 0
	could_load, checkpoint_counter = load(config.CHECKPOINT_DIR, sess)
	if could_load:
		curr_step = checkpoint_counter
		print(" [*] Load SUCCESS")
	else:
		print(" [!] Load failed...")

	for e in range(EPOCHS):
		for _ in range(ITERATIONS):

			instance_noise_std_value = get_instance_noise_std(curr_step)

			x = get_next_image_batch(config.BATCH_SIZE)
			x = normalize_image_batch(x)

			rand = np.random.uniform(0., 1., size=[config.BATCH_SIZE, 100]).astype(np.float32)
			feed_dict = {X: x, z: rand, instance_noise_std: instance_noise_std_value}
			_, D_loss_curr = sess.run([disc_optimizer, D_loss], feed_dict)

			# Run generator twice
			_, G_loss_curr = sess.run([generator_optimizer, G_loss], feed_dict)
			summary, _, G_loss_curr = sess.run([merged, generator_optimizer, G_loss], feed_dict)

			sys.stdout.write("\rstep %d: %f, %f" % (curr_step, D_loss_curr, G_loss_curr))
			sys.stdout.flush()

			curr_step += 1

			if curr_step > 0 and curr_step % STEPS_PER_IMAGE_SAMPLE == 0:
				# Note that these samples have "pixels" in the range (-1,1)
				generated_samples = sess.run(G, {z: rand})
				save_samples(generated_samples, curr_step / STEPS_PER_IMAGE_SAMPLE)

			if curr_step > 0 and curr_step % STEPS_PER_SAVE == 0:
				save(config.CHECKPOINT_DIR, curr_step, sess)

			if curr_step > 0 and curr_step % STEPS_PER_SUMMARY == 0:
				train_writer.add_summary(summary, curr_step)

