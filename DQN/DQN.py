import traceback
import copy

import numpy as np
import tensorflow as tf
from tensorflow import keras

import gym
from baselines.common.atari_wrappers import wrap_deepmind

memory_cap = 100000
output_freq = 10
batch_size = 64
frames_init = 1e4
total_episodes = 1e5
gamma = 0.99
e_max = 1.
e_min = 0.01
lam = 1e-3

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

def gen_samples(model, sars_tuples, gamma):
    default = np.zeros(env.observation_space.shape)
    targets = np.zeros((len(sars_tuples), env.action_space.n))

    terminal = [1 if s is None else 0 for s in sars_tuples[:,-1]]
    S_b = np.array([s for s in sars_tuples[:,0]])
    Sb = np.array([s if s is not None else default for s in sars_tuples[:,-1]])

    S_targets = model.predict(S_b)
    Stargets = model.predict(Sb)

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

if load_model:
    model = keras.models.load_model(model_path)
else:
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(env.action_space.n, activation='linear')
    ])

sgd = keras.optimizers.RMSprop(learning_rate=0.00025)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['acc'])


max_avg = 0.
time_steps= 0.
rewards = []
bank = memory(N = memory_cap)

for episode in range(int(total_episodes)):
    s = env.reset()
    R = 0

    for t in range(10000):
        if load_model == False:
            epsilon = e_min + (e_max - e_min)*np.exp(-lam*(time_steps-frames_init))
        else:
            epsilon = 0.01
            env.render()

        if np.random.randn() < epsilon or len(bank.memory) < frames_init:
            a = env.action_space.sample()
        else:
            a = np.argmax(model.predict(np.expand_dims(s, axis=0)))

        s_ = copy.deepcopy(s)
        s, r, done, info = env.step(a)      
        
        if done:
            s = None
            
        R += r
        time_steps += 1

        sars = [s_, a, r, s]
        bank.add_memory(sars)
        
        if len(bank.memory) > frames_init:
            sars_tuples = bank.sample_memory(n=batch_size)
            states, targets = gen_samples(model, sars_tuples, gamma)
            model.fit(states, targets, verbose=0)
        
        if done:
            break

    rewards.append(R)
    if episode % output_freq == 0 and episode > 0 and load_model == False:
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
