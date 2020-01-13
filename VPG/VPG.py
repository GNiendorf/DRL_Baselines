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

layer = keras.layers.Dense(64, activation=tf.nn.relu)
layer2 = keras.layers.Dense(64, activation=tf.nn.relu)
out = keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax)
model = keras.Sequential([layer, layer2, out])
optimizer = tf.optimizers.Adam(learning_rate = .0001)

bank = memory()
gamma = 0.99
max_time = 10000
time = 0

uniq_id = "./tensorboard3/"+'{0:%Y-%m-%d--%H:%M:%S}'.format(datetime.datetime.now())
writer = tf.summary.create_file_writer(uniq_id)

while time < max_time: 
    R = 0
    s_ = env.reset()
    done = False
    while not done:
        probs = model.predict(s_[None])[0]
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

    with tf.GradientTape() as tape:
        batch_probs = model(sars[0].astype(np.float32))
        acts = sars[1]
        acts = tf.cast(acts, tf.float32)
        pis = tf.multiply(batch_probs, acts)
        pis = tf.reduce_sum(pis, axis=1)
        returns = tf.cast(returns, tf.float32)
        returns = tf.reshape(returns, (returns.shape[0],))
        loss = tf.math.multiply(tf.math.log(pis), tf.stop_gradient(returns))
        loss = -tf.reduce_mean(loss)
        with writer.as_default():
            tf.summary.scalar("loss", loss, step=time)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    bank.clear_memory()
    time += 1
    if time%(100) == 0:
        print(time)
