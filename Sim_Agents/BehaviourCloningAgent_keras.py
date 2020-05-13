import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import adam, sgd
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint

import numpy as np
import sys
import csv
import gym
import os
import gym_SmartLoader.envs
from stable_baselines import SAC
from stable_baselines import logger
import random

def imitation_learning(buffer, env_id, nn_size, batch_size, lr, epochs, evaluations, new_model = True, train = True ):

    observations = buffer['st']
    observations = np.array(observations)
    ob_size = observations.shape[-1]

    labels = buffer['lb']
    labels = np.array(labels)

    ac_size = labels.shape[-1]

    if new_model:   ## create new sequential keras model
        print("TensorFlow version: ", tf.__version__)

        model = Sequential()

        ## input layer
        model.add(Dense(nn_size[0], input_dim=ob_size,  activation='relu'))
        ## hidden layers:
        for i in range(len(nn_size)-1):
            model.add(Dense(nn_size[i+1], activation='relu'))
        ## output later
        model.add(Dense(ac_size, activation='tanh', use_bias=False))

        print(model.summary())

        # sgd = keras.optimizers.SGD(learning_rate=1e-4)

        opt = adam(learning_rate=lr)

        model.compile(
            loss='mean_squared_error',
            optimizer='adam')

        test_name = 'keras_test_1'
        log_dir = '/home/graphics/git/SmartLoader/log_dir/' + test_name + '/'
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    else:   ### load existing sequential model
       model = load_model('/home/graphics/git/SmartLoader/saved_models/Push/1_rock/BC_40_ep_3_hist_reduced_ss')

    if train:   ## train new agent
        model.fit(
            x=observations,
            y=labels,
            batch_size=batch_size,
            verbose=2,
            epochs=epochs
        )
        model.save('/home/graphics/git/SmartLoader/saved_models/Push/test_model')

    print(' ------------ now lets compare -------------')

    env = gym.make(env_id).unwrapped
    ob_size = env.observation_space.shape[-1]

    for _ in range(evaluations):
        obs = env.reset()
        done = False
        while not done:
            act = model.predict(obs.reshape([1,ob_size]))
            obs, reward, done, info = env.step(act[0])

    env.close()


###########  labels[550]  #####  model.predict(states[550].reshape([1,ob_size]))
def main():

    mission = 'PushStonesEnv'  # Change according to algorithm
    env_id = mission + '-v0'
    env = gym.make(env_id).unwrapped

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    expert_path = '/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/40_ep_3_hist_reduced_ss/'

    states = np.load(expert_path + 'obs.npz')['arr_0']
    actions = np.load(expert_path + 'act.npy')
    starts = np.load(expert_path + 'ep_str.npy')

    new_actions = np.delete(actions, np.argwhere(starts), axis=0)

    dones=np.zeros(starts.shape)
    dones[-1]=1
    dones[np.delete(np.argwhere(starts),0)+1]=1




    # new_start = starts[1:len(labels)]

    replay_buffer = {"st": states, "lb": actions}

    nn_size = [1024, 512]
    # nn_size = [256, 256]
    batch_size = 128
    learning_rate = 1e-4
    epochs = 2000
    evaluations = 50

    imitation_learning(replay_buffer, env_id, nn_size, batch_size, learning_rate, epochs,
                       evaluations)

if __name__ == '__main__':
    main()