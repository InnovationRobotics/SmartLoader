import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Conv1D, Conv2D, Conv3D, MaxPooling2D, Flatten, BatchNormalization, concatenate
from keras.optimizers import Adam, sgd
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model

import numpy as np
import sys
import csv
import gym
import os
import gym_SmartLoader.envs
from stable_baselines import SAC
from stable_baselines import logger
import random
import pickle

def imitation_learning(states, labels, env_id, nn_size, conv_layers, batch_size, lr, epochs, evaluations, new_model = True, train = True ):

    ob_size = states.shape[1:3]

    ac_size = labels.shape[-1]

    env = gym.make(env_id).unwrapped

    assert ob_size == env.observation_space.shape, "observation space mismatch. expert data: {}, gym enviornment: {}".format(ob_size, env.observation_space.shape)
    assert ac_size == env.action_space.shape[0], "action space mismatch. expert data: {}; gym enviornment: {}".format(ac_size, env.action_space.shape)

    heat_map = states[:,:,0:-1].reshape(np.hstack([states[:,:,0:-1].shape, 1]))
    hmap_size = heat_map.shape


    sensors = states[:,0:12,-1]
    sns_shape = sensors.shape

    if new_model:   ## create new sequential keras model

        hmap_input = Input(shape=heat_map.shape[1:4], name='Heat_map_input')
        sens_input = Input(shape=(sensors.shape[-1],), name='IMU_input')

        conv_l=hmap_input

        l_strides = (1, 1)
        # for layer in range(conv_layers):
        #     # if ((layer + 1) % ((conv_layers - 2) / 2)) == 0:
        #     #     l_strides = (2, 1)
        #     # else:
        #     #     l_strides = (1, 1)
        #     conv_l = Conv2D(filters=32, kernel_size=(3, 3), strides=l_strides, padding='same', activation='relu')(conv_l)
        #     conv_l = BatchNormalization()(conv_l)
        # conv_l = Flatten()(conv_l)

        conv_l = Conv2D(filters=32, kernel_size=(3, 3), strides=l_strides, padding='same', activation='relu')(conv_l)

        # fc_l = concatenate([conv_l, sens_input])
        fc_l = conv_l
        for i in range(len(nn_size)):
            fc_l = Dense(nn_size[i], activation='relu')(fc_l)

        output = Dense(ac_size, activation='tanh', use_bias=False)(fc_l)

        model = Model(inputs=[hmap_input, sens_input], outputs=output)

        opt = Adam(lr=lr)

        model.compile(
            loss='mean_squared_error',
            optimizer=opt)

        print(model.summary())

        # test_name = 'keras_test_1'
        # log_dir = '/home/graphics/git/SmartLoader/log_dir/' + test_name + '/'
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    else:   ### load existing sequential model
       model = load_model('/home/graphics/git/SmartLoader/saved_models/Heatmap/test_model')

    if train:   ## train new agent
        model.fit(
            [heat_map,sensors],
            labels,
            batch_size=batch_size,
            verbose=1,
            epochs=epochs
        )
        model.save('/home/graphics/git/SmartLoader/saved_models/Heatmap/test_model')

    print(' ------------ now lets compare -------------')


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

    mission = 'PushStonesHeatMapEnv'  # Change according to algorithm
    env_id = mission + '-v0'

    expert_path = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/30_ep_4_st_3_hist/'

    states = np.load(expert_path + 'obs.npz')['arr_0']
    labels = np.load(expert_path + 'act.npy')

    nn_size = [256, 256, 128, 128]
    conv_layers = 3
    batch_size = 32
    learning_rate = 1e-3
    epochs = 200
    evaluations = 50

    imitation_learning(states, labels, env_id, nn_size, conv_layers, batch_size, learning_rate, epochs,
                       evaluations)

if __name__ == '__main__':
    main()