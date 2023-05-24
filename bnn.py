import tensorflow as tf
from keras import layers
import tensorflow_probability as tfp
import numpy as np
import random

tfd = tfp.distributions
tfpl = tfp.layers
tfd = tfp.distributions

def build_bnn(input_shape, num_samples):
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(num_samples, dtype=tf.float32))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tfpl.DenseVariational(128, posterior_mean_field, prior_trainable, kl_weight=1/num_samples, activation='relu'),
        tfpl.DenseVariational(32, posterior_mean_field, prior_trainable, kl_weight=1/num_samples, activation='relu'),
        tf.keras.layers.Dense(1) #regression
    ])
    return model


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])


def negative_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def compile_bnn(bnn):
    bnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanSquaredError()],
                experimental_run_tf_function=False)
    return bnn



def prepare_data(buffer, batch_size):
    state_action_pairs = []
    rewards = []
    """indx = np.random.randint(buffer.shape[0], batch_size)
    batch = buffer[indx]""" # with replacement
    batch = random.sample(buffer, batch_size)
    for experience in batch:
        state, action, reward, _ = experience
        one_hot_action = np.zeros((1,8, 8))
        one_hot_action[0][action[0]][action[1]]=1
        state_action = np.vstack((state, one_hot_action))
        state_action_pairs.append(state_action)
        rewards.append(reward)

    return np.array(state_action_pairs), np.array(rewards)



def train_bnn(model, state_action_pairs, rewards, epochs=10):
    model.fit(state_action_pairs, rewards, epochs=epochs, verbose=1) # learning rate 10^-3
