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
lr = .00025

#Model Choices
ddqn = True
target_network = True
model_path = "./atari.h5"
load_model = False

if ddqn and not target_network:
    raise Exception()

#Env Choice
atari = True
env_name = 'BreakoutDeterministic-v4'

#Assumed Hyperparameters
output_freq = 100
total_episodes = 1e6
gamma = 0.99
e_max = 1.0

if atari:
    env = wrap_deepmind(gym.make(env_name), frame_stack=True, clip_rewards=False)
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
    targets = np.zeros((len(sars_tuples[1]), env.action_space.n))

    state_preds = model.predict(sars_tuples[0])
    double_state_preds = model.predict(sars_tuples[3])
    next_state_preds = target_model.predict(sars_tuples[3])

    for idx in range(len(sars_tuples[1])):
        s_ = sars_tuples[0][idx]
        a = sars_tuples[1][idx]
        r = sars_tuples[2][idx]
        s = sars_tuples[3][idx]
        term = sars_tuples[4][idx]

        state_pred = state_preds[idx]
        double_state_pred = double_state_preds[idx]
        next_state_pred = next_state_preds[idx]
        
        double_a = np.argmax(double_state_pred)

        if term == 0:
            if ddqn:
                state_pred[a] = r + gamma*next_state_pred[double_a]
            elif target_network:
                state_pred[a] = r + gamma*np.max(next_state_pred)
            else:
                state_pred[a] = r + gamma*np.max(double_state_pred)
        else:
            state_pred[a] = r
        
        targets[idx] = state_pred
    
    return targets

def create_model():
    if atari:
        model = keras.Sequential([
            keras.layers.Conv2D(16, (8,8), strides=4, activation='relu', kernel_initializer=VarianceScaling(), input_shape=(84, 84, 4)),
            keras.layers.Conv2D(32, (4,4), strides=2, activation='relu', kernel_initializer=VarianceScaling()),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu', kernel_initializer=VarianceScaling()),
            keras.layers.Dense(env.action_space.n, activation='linear', kernel_initializer=VarianceScaling())
        ])
    else:
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(env.action_space.n, activation='linear')
        ])
    return model

if load_model:
    model = keras.models.load_model(model_path)
    target_model = keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())
else:
    model = create_model()
    target_model = keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())

target_model.compile(optimizer=RMSprop(lr=lr, rho=0.95, epsilon=0.01), loss='huber_loss', metrics=['acc'])
model.compile(optimizer=RMSprop(lr=lr, rho=0.95, epsilon=0.01), loss='huber_loss', metrics=['acc'])

max_avg = -1e4
time_steps= 0.
rewards = []
Q = []
bank = memory(N = memory_cap)

for episode in range(int(total_episodes)):
    s_ = env.reset()
    R = 0
    done = False

    while done is False:
        if load_model is False:
            epsilon = np.max([(e_min-e_max)/exp_frames*(time_steps - frames_init) + 1, e_min])
        else:
            epsilon = 0.05
            env.render()

        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            q_vals = model.predict(np.array(s_)[None])
            Q.append(np.mean(q_vals))
            a = np.argmax(q_vals)

        s, r, done, info = env.step(a)

        R += r
        time_steps += 1

        sars = [s_, a, r, s, done]
        bank.add_memory(sars)
        s_ = s

        if len(bank.memory) >= frames_init and load_model is False:
            sars_tuples = bank.sample_memory(batch_size=batch_size)
            targets = gen_targets(model, target_model, sars_tuples)
            model.fit(sars_tuples[0], targets, verbose=0, batch_size=batch_size, epochs=1)

        if time_steps % update_freq == 0. and len(bank.memory) >= frames_init and load_model is False:
            print("updating target model!")
            target_model.set_weights(model.get_weights())

    rewards.append(R)
    if episode % output_freq == 0 and episode > 0:
        np.savetxt('rewards.txt', rewards)
        print("\n")
        print("------------------------------------------")
        print("Current Epsilon: {}".format(epsilon))
        print("Episodes: {}".format(episode))
        print("Steps: {}".format(time_steps))
        avg = np.sum(rewards[-output_freq:])/output_freq
        print("Average reward for last {} episodes : {}".format(output_freq, avg))
        print("------------------------------------------")
        print("\n")
        if avg > max_avg and load_model is False:
            print("Average Reward increased from {} to {}, saving model.".format(max_avg, avg))
            model.save(model_path)
            max_avg = avg

env.close()
