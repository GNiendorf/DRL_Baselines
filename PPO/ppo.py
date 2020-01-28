import datetime

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import VarianceScaling
import tensorflow_probability as tfp
tfd = tfp.distributions
from baselines.common.atari_wrappers import wrap_deepmind

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class memory:
    def __init__(self):
        self.memory = []

    def add_memory(self, sars_tuple):
        self.memory.append(sars_tuple)

    def recall_memory(self):
        indices = np.arange(len(self.memory))
        sampled_memory = self.decode_sample(indices)
        return sampled_memory

    def decode_sample(self, indices):
        s_, a, r, s, term = [], [], [], [], []
        for i in indices:
            data = self.memory[i]
            obs_t, action, reward, obs_tp1, done = data
            s_.append(np.array(obs_t, copy=False))
            a.append(np.array(action, copy=False))
            r.append(reward)
            s.append(np.array(obs_tp1, copy=False))
            term.append(done)
        return (np.array(s_), np.array(a), np.array(r), np.array(s), np.array(term))

    def clear_memory(self):
        self.memory = []

def get_returns(sars, gamma, model_val):
    returns = []
    for idx, r in enumerate(sars[2]):
        undiscounted = sars[2][idx+1:]
        final_state_val = 0
        done_last = sars[4][-1]
        final_state = sars[3][-1]
        if not done_last:
            final_state_val = model_val.predict(final_state[None])[0][0]
        undiscounted *= gamma**np.arange(len(undiscounted))
        returns.append([np.sum(undiscounted) + final_state_val])
    returns = tf.cast(returns, tf.float32)
    return returns

def train_model(sars, gamma, model_val, model_act, true_time):
    returns = get_returns(sars, gamma, model_val)

    batch_probs_old = model_act(sars[0].astype(np.float32))
    acts = tf.cast(sars[1], tf.float32)
    pis_old = tf.multiply(batch_probs_old, acts)
    pis_old = tf.reduce_sum(pis_old, axis=1, keepdims=True)
    pis = pis_old

    for N in range(epochs):
        true_time += 1
        with tf.GradientTape() as tape:
            batch_probs_new = model_act(sars[0].astype(np.float32))
            state_vals = model_val(sars[0].astype(np.float32))
            acts = tf.cast(sars[1], tf.float32)
            pis = tf.multiply(batch_probs_new, acts)
            pis = tf.reduce_sum(pis, axis=1, keepdims=True)
            kl_approx = tf.reduce_mean(tf.square(tf.subtract(pis, pis_old)))
            ratio = tf.math.divide(pis, pis_old)
            ratio_clipped = tf.clip_by_value(ratio, 1-epsilon, 1+epsilon)
            returns_base = tf.subtract(returns, state_vals)
            state_loss = tf.square(returns_base)
            state_loss = tf.reduce_mean(state_loss)
            entropy = tf.math.abs(tf.nn.softmax_cross_entropy_with_logits(labels=acts, logits=batch_probs_new))
            entropy = tf.reshape(entropy, (entropy.shape[0],1))
            loss = tf.math.multiply(ratio, tf.stop_gradient(returns_base))
            loss_clipped = tf.math.multiply(ratio_clipped, tf.stop_gradient(returns_base))
            true_loss = tf.math.minimum(loss, loss_clipped)
            total_loss = true_loss + beta*entropy - kappa*state_loss
            total_loss = -tf.reduce_mean(total_loss)
            with writer.as_default():
                tf.summary.scalar("policy loss", total_loss, step=true_time)
                tf.summary.scalar("policy entropy", tf.reduce_mean(entropy), step=true_time)
                tf.summary.scalar("approximate KL", kl_approx, step=true_time)

        gradients_act = tape.gradient(total_loss, model_act.trainable_variables)
        optimizer.apply_gradients(zip(gradients_act, model_act.trainable_variables))

    with tf.GradientTape() as tape:
        state_vals = model_val(sars[0].astype(np.float32))
        returns_base = tf.subtract(returns, state_vals)
        state_loss = tf.square(returns_base)
        state_loss = tf.reduce_mean(state_loss)
        with writer.as_default():
            tf.summary.scalar("state loss", state_loss, step=time)

    gradients_state = tape.gradient(state_loss, model_val.trainable_variables)
    optimizer.apply_gradients(zip(gradients_state, model_val.trainable_variables))
    return true_time

env = wrap_deepmind(gym.make('BreakoutDeterministic-v4'), frame_stack=True, clip_rewards=False)
uniq_id = "./ppo_sea_fixed/"+'{0:%Y-%m-%d--%H:%M:%S}'.format(datetime.datetime.now())
writer = tf.summary.create_file_writer(uniq_id)

layer1a = keras.layers.Conv2D(16, (8,8), strides=4, activation='relu', input_shape=(84, 84, 4))
layer2a = keras.layers.Conv2D(32, (4,4), strides=2, activation='relu')
layer3a = keras.layers.Flatten()
layer4a = keras.layers.Dense(256, activation='relu')

out = keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax)
out2 = keras.layers.Dense(1, activation='linear')

model_act = keras.Sequential([layer1a, layer2a, layer3a, layer4a, out])
model_val = keras.Sequential([layer1a, layer2a, layer3a, layer4a, out2])
optimizer = tf.optimizers.Adam(learning_rate=3e-4)

bank = memory()
max_time = 1e9
time = 0

train_freq = 100
t_max = 20000
gamma = .99
beta = .1
kappa = .5
epochs = 3
epsilon = .2
true_time = 0

while time < max_time: 
    R = 0
    s_ = env.reset()
    done = False
    time_start = time
    while not done and time-time_start < t_max:
        time += 1
        probs = model_act.predict(s_[None])[0]
        a = np.random.choice(len(probs), p=probs)
        s, r, done, info = env.step(a)
        R += r
        one_hot = np.zeros(len(probs))
        one_hot[a] = 1
        sars = [s_, one_hot, r, s, done]
        bank.add_memory(sars)
        s_ = s

        if ((time-time_start) % train_freq == 0 or done) and len(bank.memory) > 1:
            sars = bank.recall_memory()
            true_time = train_model(sars, gamma, model_val, model_act, true_time)
            bank.clear_memory()

    bank.clear_memory()

    with writer.as_default():
        tf.summary.scalar('Reward', R, step=time)
