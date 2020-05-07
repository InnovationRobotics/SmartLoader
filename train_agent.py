#!/usr/bin/env python3

import os
from stable_baselines.sac.policies import MlpPolicy as sac_MlpPolicy
from stable_baselines.sac.policies import CnnPolicy
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

n_steps = 0
save_interval = 2000
best_mean_reward = -np.inf

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


def data_saver(obs, act, rew, dones, ep_rew):

    np.save('/home/graphics/git/SmartLoader/saved_ep/obs', obs)
    np.save('/home/graphics/git/SmartLoader/saved_ep/act', act)
    np.save('/home/graphics/git/SmartLoader/saved_ep/rew', rew)

    ep_str = [False] * len(dones)
    ep_str[0] = True

    for i in range(len(dones) - 1):
        if dones[i]:
            ep_str[i + 1] = True

    np.save('/home/graphics/git/SmartLoader/saved_ep/ep_str', ep_str)
    np.save('/home/graphics/git/SmartLoader/saved_ep/ep_ret', ep_rew)


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
    save_path = os.getcwd() + '/' + name + '_dataset'
    # os.makedirs(save_path)
    np.savez(save_path, **numpy_dict)

    return numpy_dict


def main():
    global model, best_model_path, last_model_path
    # mission = 'PushStonesEnv' # Change according to algorithm
    mission = 'PushAlgoryxEnv'
    # env = gym.make(mission + '-v0').unwrapped
    env = gym.make(mission + '-v0')

    # Create log and model dir
    dir = 'stable_bl/' + mission
    # dir = 'stable_bl/PushMultipleStones'
    os.makedirs(dir + '/model_dir/sac', exist_ok=True)

    jobs = ['train', 'record', 'BC_agent', 'play']
    job = jobs[0]

    name = 'PickUp_40_episodes'
    pretrain = False
    fillBuffer = False

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
        model = SAC(CnnPolicy, env, gamma=0.99, learning_rate=1e-4, buffer_size=50000,
             learning_starts=1000, train_freq=1, batch_size=64,
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

        # model = SAC.load(dir + '/model_dir/sac/test_0_11_16_2.zip',
        #                  env=env, tensorboard_log=log_dir,
        #                  custom_objects=dict(learning_starts=0)) #, learning_rate=2e-4,
        #                                      # train_freq=8, gradient_steps=4, target_update_interval=4))
        # #                                              # batch_size=32))

        # pretrain
        if pretrain:
            # load dataset only once
            # expert_dataset(name)
            dataset = ExpertDataset(expert_path=(os.getcwd() + '/' + name + '_dataset.npz'), traj_limitation=-1)
            model.pretrain(dataset, n_epochs=2000)

        # fill replay buffer with Benny's recordings
        if fillBuffer:
            traj = expert_dataset(name)

            for i in range(len(traj['obs'])-1):
                if traj['episode_starts'][i+1]:
                    done = True
                else: done = False

                obs = traj['obs'][i]
                action = traj['actions'][i]
                reward = traj['rewards'][i]
                next_obs = traj['obs'][i+1]

                model.replay_buffer.add(obs, action, reward, next_obs, float(done))

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

        # mission = 'PushStonesRecorder'  # Change according to algorithm
        mission = 'PushStonesEnv'
        env = gym.make(mission + '-v0').unwrapped

        num_episodes = 10

        obs = []
        actions = []
        rewards = []
        dones = []
        episode_rewards = []

        for episode in range(num_episodes):

            ob = env.reset()
            done = False
            print('Episode number ', episode)
            episode_reward = 0

            while not done:

                act = "recording"
                new_ob, reward, done, action = env.step(act)

                # ind = [0, 1, 2, 18, 21, 24]
                ind = [0, 1, 2]
                # print(ob)

                obs.append(ob)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                episode_reward = episode_reward + reward

                ob = new_ob

            episode_rewards.append(episode_reward)



        data_saver(obs, actions, rewards, dones, episode_rewards)

    elif job == 'play':
        # env = gym.make('PickUpEnv-v0')
        model = SAC.load(dir + '/model_dir/sac/test_0_11_16_2.zip', env=env, custom_objects=dict(learning_starts=0)) ### ADD NUM

        for _ in range(2):

            obs = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                # print('state: ', obs[0:3], 'action: ', action)


if __name__ == '__main__':
    main()
