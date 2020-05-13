#!/usr/bin/env python3

import os
from stable_baselines.sac.policies import MlpPolicy as sac_MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as ddpg_MlpPolicy
from stable_baselines.common.policies import MlpPolicy as Common_MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.gail import ExpertDataset
from stable_baselines import TRPO
from stable_baselines import DDPG
from stable_baselines import PPO1
from stable_baselines import SAC
from stable_baselines import logger
from os import system
import gym
import gym_SmartLoader.envs
import time
import numpy as np
from typing import Dict
from tempfile import TemporaryFile
import csv
from pynput import keyboard
import sys
import pickle
import hickle

n_steps = 0
save_interval = 2000
best_mean_reward = -np.inf
recorder_on = True

def save_fn(_locals, _globals):
    global model, n_steps, best_mean_reward, best_model_path, last_model_path
    if (n_steps + 1) % save_interval == 0:

        # Evaluate policy training performance
        mean_reward = round(float(np.mean(_locals['episode_rewards'][-101:-1])), 1)
        print(n_steps + 1, 'timesteps')
        print("Best mean reward: {:.2f} - Last mean reward: {:.2f}".format(best_mean_reward, mean_reward))
        # New best model, save the agent
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print("Saving new best model")
            model.save(best_model_path + '_rew_' + str(np.round(best_mean_reward, 2)))
        model.save(
            last_model_path + '_' + str(time.localtime().tm_mday) + '_' + str(time.localtime().tm_hour) + '_' + str(time.localtime().tm_min))
    n_steps += 1
    pass

def on_press(key):
    global recorder_on, sim_joy

    sim_joy = [0, 0, 0, 0]
    if key.char == 'r':
        if recorder_on == False:
            print('start recording')
            recorder_on = True
        else:
            print('stop recording')
            recorder_on = False
    # if key.char == 'w':
    #     sim_joy[1] = 1
    # if key.char == 's':
    #     sim_joy[1] = -1
    # if key.char == 'a':
    #     sim_joy[0] = -1
    # if key.char == 'd':
    #     sim_joy[0] = 1
    #
    # if key.char == 'o':
    #     sim_joy[3] = 1
    # if key.char == 'k':
    #     sim_joy[3] = -1
    # if key.char == 'p':
    #     sim_joy[2] = -1
    # if key.char == 'l':
    #     sim_joy[2] = 1

def data_saver(obs, act, rew, dones, ep_rew):

    np.savez_compressed('/home/graphics/git/SmartLoader/saved_ep/obs', obs)
    # hickle.dump(obs, '/home/graphics/git/SmartLoader/saved_ep/obs_hkl_gzip.hkl', mode='w', compression='gzip')
    # np.save('/home/graphics/git/SmartLoader/saved_ep/obs', obs)
    np.save('/home/graphics/git/SmartLoader/saved_ep/act', act)
    np.save('/home/graphics/git/SmartLoader/saved_ep/rew', rew)

    ep_str = [False] * len(dones)
    ep_str[0] = True

    for i in range(len(dones) - 1):
        if dones[i]:
            ep_str[i + 1] = True

    np.save('/home/graphics/git/SmartLoader/saved_ep/ep_str', ep_str)
    np.save('/home/graphics/git/SmartLoader/saved_ep/ep_ret', ep_rew)

def ob_to_heat_map(observation):

    h_map = np.zeros([2000, 1000])

    sens_state = observation[0:-3]

    stone_pos = observation[-3:len(observation)]

    st_ind = [100 * stone_pos[0] - 24500, 100 * stone_pos[1] - 24500]
    h_map[st_ind] = 1

    obs = dict(sensors=sens_state,
               heatmap=h_map)

    return obs


def expert_dataset(name):
    # Benny's recordings to dict
    path = os.getcwd() + '/' + name
    numpy_dict = {
        'actions': np.load(path + '/act.npy'),
        'obs': np.load(path + '/obs.npy'),
        'rewards': np.load(path + '/rew.npy'),
        'episode_returns': np.load(path + '/ep_ret.npy'),
        'episode_starts': np.load(path + '/ep_str.npy')
    } # type: Dict[str, np.ndarray]

    # for key, val in numpy_dict.items():
    #     print(key, val.shape)

    # dataset = TemporaryFile()
    save_path = os.getcwd() + '/dataset'
    os.makedirs(save_path)
    np.savez(save_path, **numpy_dict)

def main():
    global model, best_model_path, last_model_path, sim_joy
    mission = 'PushStonesEnv' # Change according to algorithm
    env = gym.make(mission + '-v0').unwrapped

    # Create log and model dir
    # dir = 'stable_bl/' + mission
    dir = 'stable_bl/PushMultipleStones'
    os.makedirs(dir + '/model_dir/sac', exist_ok=True)

    jobs = ['train', 'record', 'record-w/hm', 'BC_agent', 'play']
    job = jobs[1]
    pretrain = True

    if job == 'train':

        # create new folder
        try:
            tests = os.listdir(dir + '/model_dir/sac')
            indexes = []
            for item in tests:
                indexes.append(int(item.split('_')[1]))
            if not bool(indexes):
                k = 0
            else:
                k = max(indexes) + 1
        except FileNotFoundError:
            os.makedirs(dir + '/log_dir/sac')
            k = 0

        model_dir = os.getcwd() + '/' + dir + '/model_dir/sac/test_{}'.format(str(k))

        best_model_path = model_dir
        last_model_path = model_dir

        log_dir = dir + '/log_dir/sac/test_{}'.format(str(k))
        logger.configure(folder=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])

        num_timesteps = int(1e6)

        policy_kwargs = dict(layers=[64, 64, 64])

        # SAC - start learning from scratch
        model = SAC(sac_MlpPolicy, env, gamma=0.99, learning_rate=1e-4, buffer_size=500000,
             learning_starts=0, train_freq=1, batch_size=64,
             tau=0.01, ent_coef='auto', target_update_interval=1,
             gradient_steps=1, target_entropy='auto', action_noise=None,
             random_exploration=0.0, verbose=2, tensorboard_log=log_dir,
             _init_setup_model=True, full_tensorboard_log=True,
             seed=None, n_cpu_tf_sess=None)

        # Load best model and continue learning
        # models = os.listdir(dir + '/model_dir/sac')
        # models_rew = (model for model in models if 'rew' in model)
        # ind, reward = [], []
        # for model in models_rew:
        #     ind.append(model.split('_')[1])
        #     reward.append(model.split('_')[3])
        # best_reward = max(reward)
        # best_model_ind = reward.index(best_reward)
        # k = ind[best_model_ind]
        # model = SAC.load(dir + '/model_dir/sac/test_' + k + '_rew_' + best_reward, env=env,
        #                  custom_objects=dict(learning_starts=0))
        # Load last saved model and continue learning
        # models = os.listdir(dir + '/model_dir/sac')
        # models_time = (model for model in models if 'rew' not in model)
        # ind, hour, min = [], [], []
        # for model in models_time:
        #     ind.append(model.split('_')[1])
        #     hour.append(model.split('_')[3])
        #     min.append(model.split('_')[4])
        # date = models_time[0].split('_')[2]
        # latest_hour = max(hour)
        # latest_hour_ind = [i for i, n in enumerate(hour) if n == latest_hour]
        # latest_min = max(min[latest_hour_ind])
        # latest_min_ind = min(latest_min)
        # k = ind[latest_min_ind]
        # model = SAC.load(dir + '/model_dir/sac/test_' + k + '_' + date + '_' + latest_hour[0] + '_' + latest_min + 'zip',
        #                  env=env, custom_objects=dict(learning_starts=0))

        # model = SAC.load(dir + '/model_dir/sac/test_53_rew_24383.0',
        #                  env=env, tensorboard_log=log_dir,
        #                  custom_objects=dict(learning_starts=0, learning_rate=2e-4,
        #                                      train_freq=8, gradient_steps=4, target_update_interval=4))
        # #                                              # batch_size=32))

        # pretrain
        if pretrain:
            # load dataset only once
            # expert_dataset('3_rocks_40_episodes')
            dataset = ExpertDataset(expert_path=(os.getcwd() + '/dataset.npz'), traj_limitation=-1)
            model.pretrain(dataset, n_epochs=2000)

        # Test the pre-trained model
        # env = model.get_env()
        # obs = env.reset()
        #
        # reward_sum = 0.0
        # for _ in range(1000):
        #     action, _ = model.predict(obs)
        #     obs, reward, done, _ = env.step(action)
        #     reward_sum += reward
        #     if done:
        #         print(reward_sum)
        #         reward_sum = 0.0
        #         obs = env.reset()
        #
        # env.close()

        # learn
        model.learn(total_timesteps=num_timesteps, callback=save_fn)

        # PPO1
        # model = PPO1(Common_MlpPolicy, env, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01,
        #      optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5,
        #      schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True,
        #      policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)

        # TRPO
        # model = TRPO(MlpPolicy, env, timesteps_per_batch=4096, tensorboard_log=log_dir, verbose=1)
        # model.learn(total_timesteps=500000)
        # model.save(log_dir)

    elif job == 'record':

        mission = 'PushStonesHeatMapEnv'
        env = gym.make(mission + '-v0').unwrapped

        obs = []
        actions = []
        rewards = []
        dones = []
        episode_rewards = []

        num_episodes = 30

        listener = keyboard.Listener(
            on_press=on_press)
        listener.start()

        for episode in range(num_episodes):



            ob = env.reset()
            done = False
            print('Episode number ', episode+1)
            episode_reward = 0

            while not done:

                act = "recording"
                # act = sim_joy
                # act = [0,1,0.5]
                new_ob, reward, done, info = env.step(act)

                # print(info['action'])
                # print(ob)

                if recorder_on:
                    obs.append(ob)
                    actions.append(info['action'])
                    rewards.append(reward)
                    dones.append(done)
                    episode_reward = episode_reward + reward

                ob = new_ob

            episode_rewards.append(episode_reward)

            if info['reset reason'] == 'out of boarders' or info['reset reason'] == 'limit time steps':
                episode -= 1
            else:
                print('saving data')
                data_saver(obs, actions, rewards, dones, episode_rewards)


    elif job == 'play':
        # env = gym.make('PickUpEnv-v0')
        model = SAC.load(dir + '/model_dir/sac/test_25_25_14_15', env=env, custom_objects=dict(learning_starts=0)) ### ADD NUM

        for _ in range(2):

            obs = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                # print('state: ', obs[0:3], 'action: ', action)


if __name__ == '__main__':
    main()
