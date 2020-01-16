import datetime

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

env = gym.make('CartPole-v0')

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

layer = keras.layers.Dense(16, activation=tf.nn.relu)
out = keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax)
out2 = keras.layers.Dense(1, activation='linear')
model_act = keras.Sequential([layer, out])
model_val = keras.Sequential([layer, out2])
optimizer = tf.optimizers.RMSprop(learning_rate = .005, rho=.99)

bank = memory()
gamma = 0.99
max_time = 1e6
time = 0
beta = .01
kappa = .5

uniq_id = "./a3c_tb2/"+'{0:%Y-%m-%d--%H:%M:%S}'.format(datetime.datetime.now())
writer = tf.summary.create_file_writer(uniq_id)

while time < max_time: 
    R = 0
    s_ = env.reset()
    done = False
    while not done:
        probs = model_act.predict(s_[None])[0]
        a = np.random.choice(len(probs), p=probs)
        s, r, done, info = env.step(a)
        R += r
        one_hot = np.zeros(len(probs))
        one_hot[a] = 1
        sars = [s_, one_hot, r, s, done]
        bank.add_memory(sars)
        s_ = s

    with writer.as_default():
        tf.summary.scalar('Reward', R, step=time)
    sars = bank.recall_memory()
    returns = []
    for idx, r in enumerate(sars[2]):
        returns.append([np.sum(sars[2][idx+1:])])

    returns = tf.cast(returns, tf.float32)
    returns = tf.reshape(returns, (returns.shape[0],))

    with tf.GradientTape() as tape:
        batch_probs = model_act(sars[0].astype(np.float32))
        state_vals = model_val(sars[0].astype(np.float32))
        acts = tf.cast(sars[1], tf.float32)
        pis = tf.multiply(batch_probs, acts)
        pis = tf.reduce_sum(pis, axis=1)
        returns_base = tf.subtract(returns, state_vals)
        entropy = tf.math.multiply(tf.math.log(pis), pis)
        entropy = -tf.reduce_mean(entropy)
        loss = tf.math.multiply(tf.math.log(pis), tf.stop_gradient(returns_base))
        loss = -tf.reduce_mean(loss)
        state_loss = tf.square(returns_base)
        state_loss = tf.reduce_mean(state_loss)
        policy_loss = loss + beta*entropy - kappa*state_loss
        with writer.as_default():
            tf.summary.scalar("policy loss", policy_loss, step=time)

    gradients_act = tape.gradient(policy_loss, model_act.trainable_variables)
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

    bank.clear_memory()
    time += 1
    if time%100 == 0:
        print(time)
