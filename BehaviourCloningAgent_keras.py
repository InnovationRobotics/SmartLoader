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


def csv_writer(writer, action, observation):

    saver = []
    for ob in observation:
        saver.append(ob)
    for act in action:
        saver.append(act)

    writer.writerow(saver)

def imitation_learning(buffer, env_id, nn_size, batch_size, lr, epochs, evaluations, new_model = False, train = False):

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
       model = load_model('/home/graphics/git/SmartLoader/test_model')

    if train:   ## train new agent
        model.fit(
            x=observations,
            y=labels,
            batch_size=batch_size,
            verbose=2,
            epochs=epochs
        )
        model.save('/home/graphics/git/SmartLoader/test_model')

    print(' ------------ now lets compare -------------')

    env = gym.make(env_id).unwrapped

    for _ in range(evaluations):
        obs = env.reset()
        done = False
        while not done:
            act = model.predict(obs.reshape([1,ob_size]))
            obs, reward, done, info = env.step(act[0])

    env.close()

def main():

    mission = 'PushStonesEnv'  # Change according to algorithm
    env_id = mission + '-v0'
    env = gym.make(env_id).unwrapped

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    expert_path = '/home/graphics/git/SmartLoader/saved_experts/1_rock/np_expert/'

    states = np.load(expert_path + 'obs.npy')
    labels = np.load(expert_path + 'act.npy')

    replay_buffer = {"st": states, "lb": labels}

    nn_size = [128, 128, 128]
    batch_size = 64
    learning_rate = 1e-4
    epochs = 300
    evaluations = 50

    imitation_learning(replay_buffer, env_id, nn_size, batch_size, learning_rate, epochs,
                       evaluations)

if __name__ == '__main__':
    main()