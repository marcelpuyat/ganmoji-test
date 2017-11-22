import tensorflow as tf
from utils import *
import warnings

def gaussian_noise_layer(input_layer, std):
	noise = tf.random_normal(shape=tf.shape(input_layer), mean=0, stddev=std, dtype=tf.float32) 
	# We need to clip values to be between -1 and 1
	return tf.clip_by_value(tf.add(input_layer, noise), clip_value_min=-1, clip_value_max=1, name='add_gaussian_noise')

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

def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def SpecNorm(W, u=None, num_iters=1, update_collection=None, with_sigma=False, name='spec'):
  # Usually num_iters = 1 will be enough
  with tf.variable_scope(name):
	  W_shape = W.shape.as_list()
	  W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
	  if u is None:
	    u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
	  def power_iteration(i, u_i, v_i):
	    v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
	    u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
	    return i + 1, u_ip1, v_ip1
	  _, u_final, v_final = tf.while_loop(
	    cond=lambda i, _1, _2: i < num_iters,
	    body=power_iteration,
	    loop_vars=(tf.constant(0, dtype=tf.int32),
	               u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
	  )
	  if update_collection is None:
	    # warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
	                  # '. Please consider using a update collection instead.')
	    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
	    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
	    W_bar = W_reshaped / sigma
	    with tf.control_dependencies([u.assign(u_final)]):
	      W_bar = tf.reshape(W_bar, W_shape)
	  else:
	    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
	    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
	    W_bar = W_reshaped / sigma
	    W_bar = tf.reshape(W_bar, W_shape)
	    # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
	    # has already been collected on the first call.
	    if update_collection != 'NO_OPS':
	      tf.add_to_collection(update_collection, u.assign(u_final))
	  if with_sigma:
	    return W_bar, sigma
	  else:
	    return W_bar

def BatchNormalization(input, name='bn'):
	return tf.contrib.layers.batch_norm(input, center=True, scale=True, decay=0.9, is_training=True, updates_collections=None, epsilon=1e-5)	
	
def LeakyReLU(input, leak=0.2, name='lrelu'):
	return tf.maximum(input, leak*input, name='LeakyRelu')

def minibatch(inputs, num_kernels=32, kernel_dim=3):
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
		return tf.reduce_sum(tf.exp(-abs_diffs), 2)