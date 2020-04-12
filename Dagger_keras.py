import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import adam, sgd
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from pynput import keyboard

import numpy as np
import sys
import csv
import gym
import os
import gym_SmartLoader.envs
from stable_baselines import SAC
from stable_baselines import logger
import random

recorder_on = True

def on_press(key):
    global recorder_on
    if key.char == 'r':
        if recorder_on == False:
            print('start recording')
            recorder_on = True
        else:
            print('stop recording')
            recorder_on = False


def data_saver(obs, act, rew, dones, ep_rew):

    np.save('/home/graphics/git/SmartLoader/saved_ep_hist/obs', obs)
    np.save('/home/graphics/git/SmartLoader/saved_ep_hist/act', act)
    np.save('/home/graphics/git/SmartLoader/saved_ep_hist/rew', rew)

    starts = [False] * len(dones)
    starts[0] = True

    for i in range(len(dones) - 1):
        if dones[i]:
            starts[i + 1] = True

    np.save('/home/graphics/git/SmartLoader/saved_ep_hist/ep_str', starts)
    np.save('/home/graphics/git/SmartLoader/saved_ep_hist/ep_ret', ep_rew)


def imitation_learning(buffer, env_id, nn_size, batch_size, lr, epochs, train_sessions, evaluations, expert_sessions, new_model=False, train=False):

    global recorder_on
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    obs = []
    actions = []
    rewards = []
    dones = []
    episode_rewards = []


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
        opt = adam(learning_rate=lr)

        model.compile(
            loss='mean_squared_error',
            optimizer='adam')

        test_name = 'keras_test_1'
        log_dir = '/home/graphics/git/SmartLoader/log_dir/' + test_name + '/'
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    else:   ### load existing sequential model
       model = load_model('/home/graphics/git/SmartLoader/saved_models/1_rock_hist_model_2')

    if train:   ## train new agent

        for t_sess in range(train_sessions):

            model.fit(
                x=observations,
                y=labels,
                batch_size=batch_size,
                verbose=2,
                epochs=epochs
            )


            model.save('/home/graphics/git/SmartLoader/1_rock_hist_model_{}'.format(t_sess))

            env = gym.make(env_id).unwrapped

            for _ in range(evaluations):
                obs = env.reset()
                done = False
                while not done:
                    act = model.predict(obs.reshape([1, ob_size]))
                    obs, reward, done, info = env.step(act[0])

            for ep_num in range(expert_sessions):  ## new recordings :

                ob = env.reset()
                done = False
                print('Episode number ', ep_num, '============ START RECORDING ============')
                episode_reward = 0

                while not done:

                    act = "recording"
                    new_ob, reward, done, info = env.step(act)

                    if recorder_on:
                        obs.append(ob)
                        actions.append(info['action'])
                        rewards.append(reward)
                        dones.append(done)
                        episode_reward = episode_reward + reward

                    observations = np.concatenate((observations, ob.reshape([1, ob_size]) ))
                    labels = np.concatenate((labels, info['action'].reshape([1, ac_size])))

                    ob = new_ob

                episode_rewards.append(episode_reward)

                data_saver(obs, actions, rewards, dones, episode_rewards)

            env.close()

    print(' ------------ now lets test performance -------------')

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

    expert_path = '/home/graphics/git/SmartLoader/saved_experts/1_rock/80_ep_5_hist/'

    states = np.load(expert_path + 'obs.npy')
    labels = np.load(expert_path + 'act.npy')

    replay_buffer = {"st": states, "lb": labels}

    nn_size = [256, 128, 64]
    batch_size = 64
    learning_rate = 1e-4
    epochs = 500
    evaluations = 15
    train_sess = 1
    expert_sessions = 0

    imitation_learning(replay_buffer, env_id, nn_size, batch_size, learning_rate, epochs,
                       train_sess, evaluations, expert_sessions)


if __name__ == '__main__':
    main()