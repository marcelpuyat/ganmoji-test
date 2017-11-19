import tensorflow as tf
from ops import *
import config

def Discriminator(X, instance_noise_std, reuse=False, name='d'):
	# Architecture:
	#   DiscriminatorBeforeFullyConnectedLayer plus:
	# 	FC layer
	# 	Sigmoid
	with tf.variable_scope(name, reuse=reuse):
		D_r, D_h4, minibatch_features = DiscriminatorBeforeFullyConnectedLayer(X, instance_noise_std, reuse, name)

		# COMMENTED OUT: MINIBATCH DISCRIMINATION
		# # Apply strong dropout on minibatch features because we care less about it compared to image features
		minibatch_features_dropped_out = tf.nn.dropout(minibatch_features, 0.4)

		# # Only a bit of dropout for image features to prevent overfitting
		D_r_dropped_out = tf.nn.dropout(D_r, 0.8)

		D_5 = tf.concat([D_r_dropped_out, minibatch_features_dropped_out], 1)

		D_h6 = Dense(D_5, output_dim=1, name='dense')
		preds = tf.nn.sigmoid(D_h6, name='predictions')
		with tf.name_scope('discrim_preds'):
			variable_summaries(preds)
		return preds, D_h6, D_h4, minibatch_features

def DiscriminatorBeforeFullyConnectedLayer(X, instance_noise_std, reuse=False, name='d'):
	# Architecture:
	# 	Add noise
	# 	Conv3x3, BN, ReLU
	# 	Minibatch discrim computed (sent FC layer at the end)
	# 	Conv3x3, BN, ReLU
	# 	Conv3x3, BN, ReLU
	# 	Conv3x3, BN, ReLU

	# Decaying noise
	X = gaussian_noise_layer(X, instance_noise_std)
	if len(X.get_shape()) > 2:
		# X: -1, 32, 32, 4
		D_conv1 = Conv2d(X, output_dim=32, kernel=(3,3), name='conv1')
	else:
		D_reshaped = tf.reshape(X, [config.BATCH_SIZE, 128, 128, 4])
		D_conv1 = Conv2d(D_reshaped, output_dim=32, kernel=(3,3), name='conv1')

	D_conv1_reshaped = tf.reshape(D_conv1, [config.BATCH_SIZE, -1])
	minibatch_features = minibatch(D_conv1_reshaped)

	D_bn1 = BatchNormalization(D_conv1, name='conv_bn1')
	D_h1 = LeakyReLU(D_bn1)

	extra_layer_conv = Conv2d(D_h1, output_dim=32, kernel=(3,3), strides=(1,1), name='conv_extra')
	extra_layer_bn = BatchNormalization(extra_layer_conv, name="conv_extra_bn")
	extra_layer_relu = LeakyReLU(extra_layer_bn)

	D_conv2 = Conv2d(extra_layer_relu, output_dim=64, kernel=(3,3), name='conv2')
	D_bn2 = BatchNormalization(D_conv2, name='conv_bn2')
	D_h2 = LeakyReLU(D_bn2)
	D_conv3 = Conv2d(D_h2, output_dim=128, kernel=(3,3), name='conv3')
	D_bn3 = BatchNormalization(D_conv3, name='conv_bn3')
	D_h3 = LeakyReLU(D_bn3)
	D_conv4 = Conv2d(D_h3, output_dim=256, kernel=(3,3), name='conv4')
	D_bn4 = BatchNormalization(D_conv4, name='conv_bn4')
	D_h4 = LeakyReLU(D_bn4)

	D_r = tf.reshape(D_h4, [config.BATCH_SIZE, 16384])
	return D_r, D_conv3, minibatch_features


def Generator(z, reuse=False, name='g'):
	# Architecture:
	# 	Project to 1024*4*4 then reshape, then BN
	# 	Then deconv with stride 2, 5x5 filters into 512*8*8, then BN
	# 	Then deconv with stride 2, 5x5 filters into 256*16*16, then BN
	# 	Then deconv with stride 2, 5x5 filters into 4*32*32
	#   tanh
	with tf.variable_scope(name, reuse=reuse):

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

		G_conv4 = Deconv2d(G_h3, output_dim=128, batch_size=config.BATCH_SIZE, name='deconv3')
		G_bn4 = BatchNormalization(G_conv4, name='deconv3_bn')
		G_h4 = tf.nn.relu(G_bn4)
		with tf.name_scope('deconv3_activation'):
			variable_summaries(G_h4)

		G_conv5 = Deconv2d(G_h4, output_dim=64, batch_size=config.BATCH_SIZE, name='deconv4')
		G_bn5 = BatchNormalization(G_conv5, name='deconv4_bn')
		G_h5 = tf.nn.relu(G_bn5)
		with tf.name_scope('deconv4_activation'):
			variable_summaries(G_h5)

		G_conv6 = Deconv2d(G_h5, output_dim=4, batch_size=config.BATCH_SIZE, name='deconv5')
		G_r6 = tf.reshape(G_conv6, [config.BATCH_SIZE, 128*128*4])
		tanh_layer = tf.nn.tanh(G_r6)
		with tf.name_scope('tanh'):
			variable_summaries(tanh_layer)
		return tanh_layer

def ModeEncoder(x, name='e'):
	# Architecture:
	#	Discriminator CNN
	#   FC into z dim
	#   Sigmoid
	with tf.variable_scope(name):
		D,_,_ = DiscriminatorBeforeFullyConnectedLayer(x, 0, False, name='Encoder')
		D_h6 = Dense(D, output_dim=config.Z_DIM, name='dense')
		predicted_z = tf.nn.sigmoid(D_h6, name='predictedZ')
		with tf.name_scope('predictedZScope'):
			variable_summaries(predicted_z)
		return predicted_z

def GeneratorWithEmbeddings(z, embeddings, reuse, name='g'):
	# Architecture:
	# 	Project to 1024*4*4 then reshape, then BN
	# 	Then deconv with stride 2, 5x5 filters into 512*8*8, then BN
	# 	Then deconv with stride 2, 5x5 filters into 256*16*16, then BN
	# 	Then deconv with stride 2, 5x5 filters into 4*32*32
	#   tanh
	with tf.variable_scope(name):
		embeddings = tf.nn.dropout(embeddings, 0.4)
		z_with_embeddings = tf.concat([z, embeddings], 1)
		return Generator(z_with_embeddings, reuse, name)

def DiscriminatorWithEmbeddings(X, embeddings, instance_noise_std, reuse=False, name='d'):
	with tf.variable_scope(name, reuse=reuse):
		embeddings_with_noise = embeddings + tf.random_normal(shape=tf.shape(embeddings), mean=0, stddev=0.002, dtype=tf.float32)
		D_r, D_h3_conv, minibatch_features = DiscriminatorBeforeFullyConnectedLayer(X, instance_noise_std, reuse, name)
		D_h6_with_embeddings = tf.concat([D_r, embeddings_with_noise], 1)
		D_h6 = Dense(D_h6_with_embeddings, output_dim=1, name='dense')
		preds = tf.nn.sigmoid(D_h6, name='predictions')
		with tf.name_scope('discrim_preds'):
			variable_summaries(preds)
		return preds, D_h6, D_h3_conv, minibatch_features

def ModeEncoderWithEmbeddings(x, embeddings, name='e'):
	# Architecture:
	#	Discriminator CNN
	#   FC into z dim + embeddings dim
	#   Sigmoid
	with tf.variable_scope(name):
		D,_,_ = DiscriminatorBeforeFullyConnectedLayer(x, 0, False, name='Encoder')
		D_h6 = Dense(D, output_dim=config.Z_DIM, name='dense')
		predicted_z = tf.nn.tanh(D_h6, name='predictedZ')
		with tf.name_scope('predictedZScope'):
			variable_summaries(predicted_z)
		return predicted_z
