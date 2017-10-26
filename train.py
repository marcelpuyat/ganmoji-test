import tensorflow as tf
import numpy as np
np.set_printoptions(threshold='nan')
import sys
import warnings
warnings.simplefilter('error', UserWarning)

from ops import *
from model import *
import config
import utils

X = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, config.IMAGE_SIZE], name="real_images_input")
z = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 100], name="generator_latent_space_input")
instance_noise_std = tf.placeholder(tf.float32, shape=(), name="instance_noise_std")

G = Generator(z, 'Generator')
D_real_prob, D_real_logits, feature_matching_real, minibatch_similarity_real = Discriminator(X, instance_noise_std, False, 'Discriminator')
D_fake_prob, D_fake_logits, feature_matching_fake, minibatch_similarity_fake= Discriminator(G, instance_noise_std, True, 'Discriminator')

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

			x = utils.get_next_image_batch(config.BATCH_SIZE)
			x = utils.normalize_image_batch(x)

			rand = np.random.uniform(0., 1., size=[config.BATCH_SIZE, 100]).astype(np.float32)
			feed_dict = {X: x, z: rand, instance_noise_std: instance_noise_std_value}
			_, D_loss_curr = sess.run([disc_optimizer, D_loss], feed_dict)

			# Run generator twice
			_, G_loss_curr = sess.run([generator_optimizer, G_loss], feed_dict)
			summary, _, G_loss_curr = sess.run([merged, generator_optimizer, G_loss], feed_dict)

			sys.stdout.write("\rstep %d: %f, %f" % (curr_step, D_loss_curr, G_loss_curr))
			sys.stdout.flush()

			curr_step += 1

			if curr_step > 0 and curr_step % config.STEPS_PER_IMAGE_SAMPLE == 0:
				# Note that these samples have "pixels" in the range (-1,1)
				generated_samples = sess.run(G, {z: rand})
				utils.save_samples(generated_samples, curr_step / config.STEPS_PER_IMAGE_SAMPLE)

			if curr_step > 0 and curr_step % config.STEPS_PER_SAVE == 0:
				utils.save(config.CHECKPOINT_DIR, curr_step, sess, saver)

			if curr_step > 0 and curr_step % config.STEPS_PER_SUMMARY == 0:
				train_writer.add_summary(summary, curr_step)

