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

ITERATIONS = 50
BATCH_SIZE = 64
EPOCHS = 1000
IMAGE_SIZE = 32*32*4

image_filenames = []

def get_image_filenames():
	s = commands.getstatusoutput('ls data')
	filenames = ['data/' + string for string in s[1].split()]
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

def denormalize_image(image):
	return np.multiply(np.divide((1 + image), 2), 255)

def plot(samples, D_loss, G_loss, epoch, total):
	fig = plt.figure(figsize=(10, 5))
	# First 4 columns are images, last 4 is for the loss
	gs = gridspec.GridSpec(4, 8)
	gs.update(wspace=0.05, hspace=0.05)
	
	# Plot losses in last 4 columns
	ax = plt.subplot(gs[:, 4:])
	ax.plot(D_loss, label="discriminator's loss", color='b')
	ax.plot(G_loss, label="generator's loss", color='r')
	ax.set_xlim([0, total])
	ax.yaxis.tick_right()
	ax.legend()

	# Generate images
	for i, sample in enumerate(samples):
		# need to convert sample from range -1,1 to 0 255
		sample = denormalize_image(sample)
		if i > 4* 4 - 1:
			break
		# Plot in the left half
		ax = plt.subplot(gs[i % 4, int(i / 4)])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(32, 32, 4).astype(np.uint8))

	plt.savefig('./output/' + str(epoch + 1) + '.png')
	print('./output/' + str(epoch + 1) + '.png')
	plt.close()

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	tf.summary.histogram(var.op.name, var)

def gaussian_noise_layer(input_layer, std):
	noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
	return tf.add(input_layer, noise, 'add_gaussian_noise')

def Conv2d(input, output_dim=64, kernel=(5, 5), strides=(2, 2), stddev=0.02, name='conv_2d'):
	with tf.variable_scope(name):
		W = tf.get_variable('convW', [kernel[0], kernel[1], input.get_shape()[-1], output_dim],
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		with tf.name_scope('W'):
			variable_summaries(W)
		b = tf.get_variable('convb', [output_dim], initializer=tf.zeros_initializer())
		with tf.name_scope('b'):
			variable_summaries(b)
	    

		return tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding='SAME') + b

def Deconv2d(input, output_dim, batch_size, kernel=(5, 5), strides=(2, 2), stddev=0.02, name='deconv_2d'):
	with tf.variable_scope(name):
		W = tf.get_variable('deconvW', [kernel[0], kernel[1], output_dim, input.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=stddev))
		with tf.name_scope('W'):
			variable_summaries(W)
		b = tf.get_variable('deconvb', [output_dim], initializer=tf.zeros_initializer())
		with tf.name_scope('b'):
			variable_summaries(b)

		input_shape = input.get_shape().as_list()
		output_shape = [batch_size,
						int(input_shape[1] * strides[0]),
						int(input_shape[2] * strides[1]),
						output_dim]

		deconv = tf.nn.conv2d_transpose(input, W, output_shape=output_shape,
										strides=[1, strides[0], strides[1], 1])
	
		return deconv + b

def Dense(input, output_dim, stddev=0.02, name='dense'):
	with tf.variable_scope(name):
	    
		shape = input.get_shape()
		W = tf.get_variable('denseW', [shape[1], output_dim],
						initializer=tf.random_normal_initializer(stddev=stddev))
		with tf.name_scope('W'):
			variable_summaries(W)
		b = tf.get_variable('denseb', [output_dim],
							initializer=tf.zeros_initializer())
		with tf.name_scope('b'):
			variable_summaries(b)
	    
		return tf.matmul(input, W) + b

def BatchNormalization(input, name='bn'):
	return tf.contrib.layers.batch_norm(input, center=True, scale=True, decay=0.9, is_training=True, updates_collections=None, epsilon=1e-5)	
	
def LeakyReLU(input, leak=0.6, name='lrelu'):
	return tf.maximum(input, leak*input, name='LeakyRelu')

def minibatch(inputs, num_kernels=5, kernel_dim=3):
	with tf.variable_scope('minibatch_discrim'):
		W = tf.get_variable("W",
							shape=[inputs.get_shape()[1], num_kernels*kernel_dim],
							initializer=tf.random_normal_initializer(stddev=0.02))
		b = tf.get_variable("b",
							shape=[num_kernels*kernel_dim],
							initializer=tf.constant_initializer(0.0))
		variable_summaries(W)
		variable_summaries(b)
		x = tf.matmul(inputs, W) + b
		activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
		diffs = tf.expand_dims(activation, 3) - tf.expand_dims(
			tf.transpose(activation, [1, 2, 0]), 0)
		eps = tf.expand_dims(np.eye(int(inputs.get_shape()[0]), dtype=np.float32), 1)
		abs_diffs = tf.reduce_sum(tf.abs(diffs), 2) + eps
		minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
		return tf.concat([inputs, minibatch_features], 1)

def Discriminator(X, reuse=False, name='d'):
	# 2 conv3s, avg pool. then 2 conv3s, avg pool.
	with tf.variable_scope(name, reuse=reuse):
		X = gaussian_noise_layer(X, 0.2)
		if len(X.get_shape()) > 2:
			# X: -1, 32, 32, 4
			D_conv1 = Conv2d(X, output_dim=32, kernel=(3,3), name='conv1')
		else:
			D_reshaped = tf.reshape(X, [BATCH_SIZE, 32, 32, 4])
			D_conv1 = Conv2d(D_reshaped, output_dim=32, kernel=(3,3), name='conv1')
		D_h1 = LeakyReLU(D_conv1)

		D_conv2 = Conv2d(D_h1, output_dim=64, kernel=(3,3), name='conv2')
		D_h2 = LeakyReLU(D_conv2)

		D_h2_pooled = tf.nn.avg_pool(D_h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		D_conv3 = Conv2d(D_h2_pooled, output_dim=128, kernel=(3,3), name='conv3')
		D_h3 = LeakyReLU(D_conv3)
		D_conv4 = Conv2d(D_h3, output_dim=256, kernel=(3,3), name='conv4')
		D_h4 = LeakyReLU(D_conv4)

		D_h4_pooled = tf.nn.avg_pool(D_h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		D_r = tf.reshape(D_h4_pooled, [BATCH_SIZE, 256])
		D_h5 = tf.nn.dropout(D_r, 0.6)

		D_after_minibatch_discrim = minibatch(D_h5)

		D_h6 = Dense(D_after_minibatch_discrim, output_dim=1, name='dense')
		preds = tf.nn.sigmoid(D_h6, name='predictions')
		return preds, D_h6, D_h4

def Generator(z, name='g'):

	# Project to 1024*4*4 then reshape
	# Then deconv with stride 2, 5x5 filters into 512*8*8
	# Then deconv with stride 2, 5x5 filters into 256*16*16
	# Then deconv with stride 2, 5x5 filters into 4*32*32
	with tf.variable_scope(name):

		G_1 = Dense(z, output_dim=1024*4*4, name='dense')
		G_r1 = tf.reshape(G_1, [BATCH_SIZE, 4, 4, 1024])
		G_bn1 = BatchNormalization(G_r1, name='dense_bn')
		G_h1 = LeakyReLU(G_bn1)
		with tf.name_scope('dense_activation'):
			variable_summaries(G_h1)

		G_conv2 = Deconv2d(G_h1, output_dim=512, batch_size=BATCH_SIZE, name='deconv1')
		G_bn2 = BatchNormalization(G_conv2, name='deconv1_bn')
		G_h2 = LeakyReLU(G_bn2)
		with tf.name_scope('deconv1_activation'):
			variable_summaries(G_h2)

		G_conv3 = Deconv2d(G_h2, output_dim=256, batch_size=BATCH_SIZE, name='deconv2')
		G_bn3 = BatchNormalization(G_conv3, name='deconv2_bn')
		G_h3 = LeakyReLU(G_bn3)
		with tf.name_scope('deconv2_activation'):
			variable_summaries(G_h3)

		G_conv4 = Deconv2d(G_h3, output_dim=4, batch_size=BATCH_SIZE, name='deconv3')
		G_r4 = tf.reshape(G_conv4, [BATCH_SIZE, 32*32*4]) # -1 is for batch size
		tanh_layer = tf.nn.tanh(G_r4)
		with tf.name_scope('tanh'):
			variable_summaries(tanh_layer)
		return tanh_layer

X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE], name="real_images_input")
z = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 100], name="generator_latent_space_input")

G = Generator(z, 'Generator')
D_real_prob, D_real_logits, feature_matching_real = Discriminator(X, False, 'Discriminator')
D_fake_prob, D_fake_logits, feature_matching_fake = Discriminator(G, True, 'Discriminator')

tf.summary.histogram("d_real_prob", D_real_prob)
tf.summary.histogram("d_fake_prob", D_fake_prob)

D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=(tf.ones_like(D_real_logits) * (0.8))), name="disc_real_cross_entropy") # one sided label smoothing
D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)), name="disc_fake_cross_entropy")
D_fake_wrong = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)), name="generator_wrong_fake_cross_entropy")
feature_matching_loss = tf.divide(tf.reduce_mean(tf.nn.l2_loss(feature_matching_real - feature_matching_fake)), (float(16) * 16), name="feature_matching_loss")

D_loss = tf.add(D_real, D_fake, "disc_loss")
G_loss = tf.add(D_fake_wrong, 0.2 * feature_matching_loss, "generator_loss")

tf.summary.scalar("D_real_loss", D_real)
tf.summary.scalar("D_fake_loss", D_fake)
tf.summary.scalar("feature_matching_loss", feature_matching_loss)
tf.summary.scalar("G_loss", G_loss)

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


D_solver = train(D_loss, d_params, learning_rate=1e-3, beta1=0.5)
G_solver = train(G_loss, g_params, learning_rate=1e-3, beta1=0.5)

def normalize_image_batches(image_batches):
	normalized_batches = np.zeros(image_batches.shape)
	for idx, batch in enumerate(image_batches):
		normalized_batches[idx] = np.multiply(2, np.divide(batch, float(255))) - 1
	return normalized_batches

with tf.Session() as sess:
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter('tensorboard/',
										 sess.graph)

	sess.run(tf.global_variables_initializer())

	D_loss_vals = []
	G_loss_vals = []

	for e in range(EPOCHS):

		for i in range(ITERATIONS):
			x = get_next_image_batch(BATCH_SIZE)
			x = normalize_image_batches(x)

			rand = np.random.uniform(0., 1., size=[BATCH_SIZE, 100]).astype(np.float32)
			_, D_loss_curr = sess.run([D_solver, D_loss], {X: x, z: rand})

			_, G_loss_curr = sess.run([G_solver, G_loss], {X: x, z: rand})
			summary, _, G_loss_curr = sess.run([merged, G_solver, G_loss], {X: x, z: rand})

			generated_images = sess.run(G, {z: rand})

			train_writer.add_summary(summary, e*ITERATIONS + i + 1)

			D_loss_vals.append(D_loss_curr)
			G_loss_vals.append(G_loss_curr)

			sys.stdout.write("\r%d / %d: %f, %f" % (i, ITERATIONS, D_loss_curr, G_loss_curr))
			sys.stdout.flush()

		data = sess.run(G, {z: rand})
		plot(data, D_loss_vals, G_loss_vals, e, EPOCHS * ITERATIONS)
