import traceback
import copy

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import *

import gym
from baselines.common.atari_wrappers import wrap_deepmind

memory_cap = 100000
output_freq = 10
batch_size = 64
frames_init = 1e4
total_episodes = 1e3
gamma = 0.99
e_max = 1.
e_min = 0.01
exp_frames = 5e3
update_freq = 1e3

env = gym.make("CartPole-v0")
model_path = "./cartpole.h5"
load_model = False

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
        
    def sample_memory(self, n):
        indices = np.random.choice(len(self.memory), size=np.min([n, len(self.memory)]))
        return np.array([self.memory[idx] for idx in indices])

def gen_samples(model, target_model, sars_tuples, gamma):
    default = np.zeros(env.observation_space.shape)
    targets = np.zeros((len(sars_tuples), env.action_space.n))

    terminal = [1 if s is None else 0 for s in sars_tuples[:,-1]]
    S_b = np.array([s for s in sars_tuples[:,0]])
    Sb = np.array([s if s is not None else default for s in sars_tuples[:,-1]])

    S_targets = model.predict(S_b)
    Stargets = target_model.predict(Sb)

    for idx in range(len(sars_tuples)):
        s_ = S_b[idx]
        a = sars_tuples[idx,1]
        r = sars_tuples[idx,2]
        s = Sb[idx]
        term = terminal[idx]
        
        tar = S_targets[idx]
        if term == 0:
            target = Stargets[idx]
            max_target = np.max(target)
            tar[a] = r + gamma*max_target
        else:
            tar[a] = r
        
        targets[idx] = tar
    
    return S_b, targets

def create_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(env.action_space.n, activation='linear')
    ])
    return model

if load_model:
    model = keras.models.load_model(model_path)
    target_model = keras.models.load_model(model_path)
else:
    target_model = create_model()
    model = create_model()

target_model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['acc'])
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['acc'])

max_avg = 0.
time_steps= 0.
rewards = []
bank = memory(N = memory_cap)
Q_vals = []

for episode in range(int(total_episodes)):
    s = env.reset()
    R = 0

    for t in range(10000):
        if load_model == False:
            epsilon = np.max([(e_min-e_max)/exp_frames*(time_steps - frames_init) + 1, e_min])
        else:
            epsilon = 0.01
            env.render()

        if np.random.randn() < epsilon or (len(bank.memory) < frames_init and load_model == False):
            a = env.action_space.sample()
        else:
            q_vals = model.predict(np.expand_dims(s, axis=0))
            Q_vals.append(np.mean(q_vals))
            a = np.argmax(q_vals)

        s_ = copy.deepcopy(s)
        s, r, done, info = env.step(a)      
        
        if done:
            s = None
            
        R += r
        time_steps += 1

        sars = [s_, a, r, s]
        bank.add_memory(sars)
        
        if len(bank.memory) > frames_init and load_model == False:
            sars_tuples = bank.sample_memory(n=batch_size)
            states, targets = gen_samples(model, target_model, sars_tuples, gamma)
            model.fit(states, targets, verbose=0)

        if time_steps % update_freq == 0 and time_steps > frames_init and load_model == False:
            print("updating target model!")
            target_model.set_weights(model.get_weights())
        
        if done:
            break

    rewards.append(R)
    if episode % output_freq == 0 and episode > 0 and load_model == False:
        np.savetxt('qvals.txt', Q_vals)
        np.savetxt('rewards.txt', rewards)
        print("\n")
        print("------------------------------------------")
        print("Current Epsilon: %.2f" % epsilon)
        print("Episodes: ", episode)
        print("Steps: %.2E" % time_steps)
        avg = np.sum(rewards[-output_freq:])/output_freq
        print("Average reward for last %.0f episodes : %.2f" % (output_freq, avg))
        print("------------------------------------------")
        print("\n")
        if avg > max_avg:
            print("Average Reward increased from %s to %s, saving model." % (max_avg, avg))
            model.save(model_path)
            max_avg = avg

env.close()
