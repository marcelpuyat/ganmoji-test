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

IMAGE_SIZE = 64*64*4
image_filenames = []

def get_image_filenames():
	s = commands.getstatusoutput('ls data')
	filenames = ['data/' + string for string in s[1].split()]
	shuffle(filenames)
	return filenames
image_filenames = get_image_filenames() # Load all image filenames into memory

def get_pixels_for_filename(filename):
    img = scipy.misc.imread(filename, mode='RGBA')
    img = Image.fromarray(img)
    return np.array(img.getdata())

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
			if pix.shape != (4096, 4):
				print('Invalid pixels shape for file ' + image_filenames[curr_image_idx] + ': ' + str(pix.shape))
				# Skip this image
				curr_image_idx += 1
				continue

			batch[i] = get_pixels_for_filename(image_filenames[curr_image_idx]).reshape([IMAGE_SIZE])
			curr_image_idx += 1
			break
	return batch

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
		sample = np.multiply(np.divide((1 + sample), 2), 255)
		if i > 4* 4 - 1:
			break
		# Plot in the left half
		ax = plt.subplot(gs[i % 4, int(i / 4)])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(64, 64, 4))

	plt.savefig('./output/' + str(epoch + 1) + '.png')
	print('./output/' + str(epoch + 1) + '.png')
	plt.close()

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

def Conv2d(input, output_dim=64, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='conv_2d'):

	with tf.variable_scope(name):
		W = tf.get_variable('Conv2dW', [kernel[0], kernel[1], input.get_shape()[-1], output_dim],
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		with tf.name_scope('W'):
			variable_summaries(W)
		b = tf.get_variable('Conv2db', [output_dim], initializer=tf.zeros_initializer())
		with tf.name_scope('b'):
			variable_summaries(b)
	    

		return tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding='SAME') + b

def Deconv2d(input, output_dim, batch_size, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='deconv_2d'):
	
	with tf.variable_scope(name):
		W = tf.get_variable('Deconv2dW', [kernel[0], kernel[1], output_dim, input.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=stddev))
		with tf.name_scope('W'):
			variable_summaries(W)
		b = tf.get_variable('Deconv2db', [output_dim], initializer=tf.zeros_initializer())
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
		W = tf.get_variable('DenseW', [shape[1], output_dim],
						initializer=tf.random_normal_initializer(stddev=stddev))
		with tf.name_scope('W'):
			variable_summaries(W)
		b = tf.get_variable('Denseb', [output_dim],
							initializer=tf.zeros_initializer())
		with tf.name_scope('b'):
			variable_summaries(b)
	    
		return tf.matmul(input, W) + b

def BatchNormalization(input, name='bn'):
	
	with tf.variable_scope(name):
	
		output_dim = input.get_shape()[-1]
		beta = tf.get_variable('BnBeta', [output_dim],
								initializer=tf.zeros_initializer())
		gamma = tf.get_variable('BnGamma', [output_dim],
								initializer=tf.ones_initializer())
	
		if len(input.get_shape()) == 2:
			mean, var = tf.nn.moments(input, [0])
		else:
			mean, var = tf.nn.moments(input, [0, 1, 2])
		return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-5)
	
def LeakyReLU(input, leak=0.2, name='lrelu'):
	
	return tf.maximum(input, leak*input)

BATCH_SIZE = 64
EPOCHS = 1000

def Discriminator(X, reuse=False, name='d'):
	# 2 conv3s, avg pool. then 2 conv3s, avg pool.
	with tf.variable_scope(name, reuse=reuse):
		if len(X.get_shape()) > 2:
			# X: -1, 64, 64, 4
			D_conv1 = Conv2d(X, output_dim=32, kernel=(3,3), name='Disc_conv1')
		else:
			D_reshaped = tf.reshape(X, [-1, 64, 64, 4])
			D_conv1 = Conv2d(D_reshaped, output_dim=32, kernel=(3,3), name='Disc_conv1')
		D_h1 = LeakyReLU(D_conv1)
		D_conv2 = Conv2d(D_h1, output_dim=64, kernel=(3,3), name='Disc_conv2')
		D_h2 = LeakyReLU(D_conv2)

		D_h2_pooled = tf.nn.avg_pool(D_h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		D_conv3 = Conv2d(D_h2_pooled, output_dim=128, kernel=(3,3), name='Disc_conv3')
		D_h3 = LeakyReLU(D_conv3)
		D_conv4 = Conv2d(D_h3, output_dim=256, kernel=(3,3), name='Disc_conv4')
		D_h4 = LeakyReLU(D_conv4)

		D_h4_pooled = tf.nn.avg_pool(D_h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		D_r = tf.reshape(D_h4_pooled, [-1, 256])
		D_h5 = tf.nn.dropout(D_r, 0.7)
		D_h6 = Dense(D_h5, output_dim=1, name='Disc_h6')
		return tf.nn.sigmoid(D_h6), D_h6

def Generator(z, name='g'):

	# Project to 1024*4*4 then reshape
	# Then deconv with stride 2, 5x5 filters into 512*8*8
	# Then deconv with stride 2, 5x5 filters into 256*16*16
	# Then deconv with stride 2, 5x5 filters into 128*32*32
	# Then deconv with stride 2, 5x5 filters into 4*64*64
	with tf.variable_scope(name):

		G_1 = Dense(z, output_dim=1024*4*4, name='Gen_1')
		G_bn1 = BatchNormalization(G_1, name='Gen_bn1')
		G_h1 = tf.nn.relu(G_bn1)
		G_r1 = tf.reshape(G_h1, [-1, 4, 4, 1024])

		G_conv2 = Deconv2d(G_r1, output_dim=512, batch_size=BATCH_SIZE, name='Gen_conv2')
		G_bn2 = BatchNormalization(G_conv2, name='Gen_bn2')
		G_h2 = tf.nn.relu(G_bn2)

		G_conv3 = Deconv2d(G_h2, output_dim=256, batch_size=BATCH_SIZE, name='Gen_conv3')
		G_bn3 = BatchNormalization(G_conv3, name='Gen_bn3')
		G_h3 = tf.nn.relu(G_bn3)

		G_conv4 = Deconv2d(G_h3, output_dim=128, batch_size=BATCH_SIZE, name='Gen_conv4')
		G_bn4 = BatchNormalization(G_conv4, name='Gen_bn4')
		G_h4 = tf.nn.relu(G_bn4)

		G_conv5 = Deconv2d(G_h4, output_dim=4, batch_size=BATCH_SIZE, name='Gen_conv5')
		G_bn5 = BatchNormalization(G_conv5, name='Gen_bn5')
		G_h5 = tf.nn.relu(G_bn5)
		G_r5 = tf.reshape(G_h5, [-1, 64*64*4]) # -1 is for batch size
		return tf.nn.tanh(G_r5)

X = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
z = tf.placeholder(tf.float32, shape=[None, 100])

G = Generator(z, 'Generator')
_, D_real_logits = Discriminator(X, False, 'Discriminator')
_, D_fake_logits = Discriminator(G, True, 'Discriminator')

D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=(tf.ones_like(D_real_logits) * (0.9)))) # one sided label smoothing
D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))

tf.summary.scalar("D_real", D_real)
tf.summary.scalar("D_fake", D_fake)
D_loss = D_real + D_fake
G_loss = -D_fake

vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('Discriminator/')]
g_params = [v for v in vars if v.name.startswith('Generator/')]

D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.1).minimize(D_loss, var_list=d_params)
G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.3).minimize(G_loss, var_list=g_params)

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

	iteration = 4
	for e in range(EPOCHS):

		for i in range(iteration):
			x = get_next_image_batch(BATCH_SIZE)
			x = normalize_image_batches(x)
			rand = np.random.uniform(0., 1., size=[BATCH_SIZE, 100])
			summary, _, D_loss_curr = sess.run([merged, D_solver, D_loss], {X: x, z: rand})
			train_writer.add_summary(summary, i)
			rand = np.random.uniform(0., 1., size=[BATCH_SIZE, 100])

			for _ in range(5):
				_, G_loss_curr = sess.run([G_solver, G_loss], {z: rand})

			D_loss_vals.append(D_loss_curr)
			G_loss_vals.append(G_loss_curr)

			sys.stdout.write("\r%d / %d: %f, %f" % (i, iteration, D_loss_curr, G_loss_curr))
			sys.stdout.flush()

		data = sess.run(G, {z: rand})
		plot(data, D_loss_vals, G_loss_vals, e, EPOCHS * iteration)