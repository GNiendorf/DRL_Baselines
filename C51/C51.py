import datetime
import math

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import RMSprop
import gym
from baselines.common.atari_wrappers import wrap_deepmind

#Tunable Hyperparameters
memory_cap = 50000
exp_frames = 50000
update_freq = 1000
frames_init = 50000
batch_size = 32
e_min = 0.1
lr = .001
V_min = -10
V_max = 10
#Number of atoms for each action
num_atoms = 51

#Env Choice
atari = True
model_path = 'test.h5'
env_name = 'BreakoutDeterministic-v4'

#Assumed Hyperparameters
output_freq = 10
total_time = 2e8
gamma = .99
e_max = 1.0

if atari:
    env = wrap_deepmind(gym.make(env_name), frame_stack=True, clip_rewards=True)
else:
    env = gym.make(env_name)

class memory:
    def __init__(self, N):
        self.N = N
        self.memory = []
        self.next_idx = 0
        
    def add_memory(self, sars_tuple):
        if self.next_idx >= len(self.memory):
            self.memory.append(sars_tuple)
        else:
            self.memory[self.next_idx] = sars_tuple
        self.next_idx = (self.next_idx + 1) % self.N
        
    def sample_memory(self, batch_size):
        indices = np.random.choice(len(self.memory), size=np.min([batch_size, len(self.memory)]))
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

def gen_targets(model, target_model, sars_tuples):
    targets = np.zeros((len(sars_tuples[1]), env.action_space.n, num_atoms))

    state_preds = model.predict(sars_tuples[0])
    next_state_preds = target_model.predict(sars_tuples[3])

    for idx in range(len(sars_tuples[1])):
        a = sars_tuples[1][idx]
        r = sars_tuples[2][idx]
        term = sars_tuples[4][idx]

        state_pred = state_preds[idx]
        next_state_pred = next_state_preds[idx]

        q_next_state = np.sum(next_state_pred*z_vals, axis=1)
        a_star = np.argmax(q_next_state)
        next_state_pred_a = next_state_pred[a_star]

        m = np.zeros(num_atoms)
        
        if term:
            T_z_j = max(min(r, V_max), V_min)
            b_j = (T_z_j - V_min)/del_z
            u = math.ceil(b_j); l = math.floor(b_j)
            if u==l:
                m[l] += 1.
            else:
                m[l] += (u-b_j)
                m[u] += (b_j-l)
        else:
            for j in range(num_atoms):
                T_z_j = max(min(r + gamma*z_vals[j], V_max), V_min)
                b_j = (T_z_j - V_min)/del_z
                u = math.ceil(b_j); l = math.floor(b_j)
                if u==l:
                    m[l] += next_state_pred_a[j]
                else:
                    m[l] += (u-b_j)*next_state_pred_a[j]
                    m[u] += (b_j-l)*next_state_pred_a[j]

        state_pred[a] = m
        targets[idx] = state_pred
    
    return targets

def create_model():
    if atari:
        model = keras.Sequential([
            keras.layers.Conv2D(16, (8,8), strides=4, activation='relu', kernel_initializer=VarianceScaling(), input_shape=(84, 84, 4)),
            keras.layers.Conv2D(32, (4,4), strides=2, activation='relu', kernel_initializer=VarianceScaling()),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu', kernel_initializer=VarianceScaling()),
            keras.layers.Dense(env.action_space.n*num_atoms, activation='linear', kernel_initializer=VarianceScaling()),
            keras.layers.Reshape(target_shape=[-1, num_atoms]),
            keras.layers.Softmax()
        ])
    else:
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=env.observation_space.shape),
            keras.layers.Dense(env.action_space.n*num_atoms, activation='linear'),
            keras.layers.Reshape(target_shape=[-1, num_atoms]),
            keras.layers.Softmax()
        ])
    return model

model = create_model()
target_model = keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

target_model.compile(optimizer=RMSprop(lr=lr, rho=0.95, epsilon=0.01), loss='categorical_crossentropy', metrics=['acc'])
model.compile(optimizer=RMSprop(lr=lr, rho=0.95, epsilon=0.01), loss='categorical_crossentropy', metrics=['acc'])

max_avg = -1e4
del_z = (V_max - V_min)/(num_atoms-1)
z_vals = np.array([V_min + i*del_z for i in range(num_atoms)])
print("Z values: {}".format(z_vals))
time_steps= 0.
episodes = 0.
rewards = []
Q = []
bank = memory(N = memory_cap)

while time_steps < total_time:
    s_ = env.reset()
    R = 0
    done = False

    while done is False:
        epsilon = np.max([(e_min-e_max)/exp_frames*(time_steps - frames_init) + 1, e_min])

        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            p_vals = model.predict(np.array(s_)[None])[0]
            q_vals = np.sum(p_vals*z_vals, axis=1)
            Q.append(np.mean(q_vals))
            a = np.argmax(q_vals)

        s, r, done, info = env.step(a)

        R += r
        time_steps += 1

        sars = [s_, a, r, s, done]
        bank.add_memory(sars)
        s_ = s

        if len(bank.memory) >= frames_init:
            sars_tuples = bank.sample_memory(batch_size=batch_size)
            targets = gen_targets(model, target_model, sars_tuples)
            model.fit(sars_tuples[0], targets, verbose=0, batch_size=batch_size, epochs=1)

        if time_steps % update_freq == 0. and len(bank.memory) >= frames_init:
            print("updating target model!")
            target_model.set_weights(model.get_weights())

    episodes += 1
    rewards.append(R)
    if episodes % output_freq == 0:
        print("\n")
        print("------------------------------------------")
        print("Current Epsilon: {}".format(epsilon))
        print("Episodes: {}".format(episodes))
        print("Steps: {}".format(time_steps))
        avg = round(np.sum(rewards[-output_freq:])/output_freq,4)
        print("Average reward for last {} episodes : {}".format(output_freq, avg))
        print("------------------------------------------")
        print("\n")
        if avg > max_avg:
            print("Average Reward increased from {} to {}, saving model.".format(max_avg, avg))
            model.save(model_path)
            max_avg = avg

env.close()
