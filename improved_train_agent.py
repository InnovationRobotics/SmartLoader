#!/usr/bin/env python3
import argparse
import os
from stable_baselines.gail import ExpertDataset
from stable_baselines import TRPO, A2C, DDPG, PPO1, PPO2, SAC, ACER, ACKTR, GAIL, DQN, HER, TD3, logger
import gym

import time
import numpy as np
import tensorflow as tf
from typing import Dict
#from tensor_board_cb import TensorboardCallback
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines.common.evaluation import evaluate_policy
import gym_SmartLoader.envs


from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn

# for custom callbacks stable-baselines should be upgraded using -
# pip3 install stable-baselines[mpi] --upgrade
from stable_baselines.common.callbacks import BaseCallback

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo1': PPO1,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3,
    'gail': GAIL
}
JOBS = ['train', 'record', 'BC_agent', 'play']

POLICIES = ['MlpPolicy', 'CnnPolicy','CnnMlpPolicy']

BEST_MODELS_NUM = 0





# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
class CnnMlpPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CnnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            feature_num = 7
            h = 291
            w = 150
            proc_obs_float = tf.cast(self.processed_obs, dtype=tf.float32)
            grid_map_flat = tf.slice(proc_obs_float,[0,0],[1,43650] )
            grid_map = tf.reshape(grid_map_flat, [1,h,w,1])
            # kwargs['data_format']='NCHW'
            extracted_features = nature_cnn(grid_map, **kwargs)
            extracted_features = tf.layers.flatten(extracted_features)
            features =  tf.slice(proc_obs_float, [0, 43650], [1, feature_num])
            pi_h = tf.concat([extracted_features,features], 1)
            for i, layer_size in enumerate([128, 128, 128]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = tf.concat([extracted_features,features], 1)
            for i, layer_size in enumerate([32, 32]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    # def _custom_nature_cnn(self):
    #     activ = tf.nn.relu
    #     layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), dataNHWC **kwargs))
    #     layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    #     layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    #     layer_3 = conv_to_fc(layer_3)
    #     return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


def expert_dataset(name):
    # Benny's recordings to dict
    path = os.getcwd() + '/' + name
    numpy_dict = {
        'actions': np.load(path + '/act.npy'),
        'obs': np.load(path + '/obs.npy'),
        'rewards': np.load(path + '/rew.npy'),
        'episode_returns': np.load(path + '/ep_ret.npy'),
        'episode_starts': np.load(path + '/ep_str.npy')
    }  # type: Dict[str, np.ndarray]

    # for key, val in numpy_dict.items():
    #     print(key, val.shape)

    # dataset = TemporaryFile()
    save_path = os.getcwd() + '/dataset'
    os.makedirs(save_path)
    np.savez(save_path, **numpy_dict)

class ExpertDatasetLoader:
    dataset = None

    def __call__(self, force_load=False):
        if ExpertDatasetLoader.dataset is None or force_load:
            print('loading expert dataset')
            ExpertDatasetLoader.dataset = ExpertDataset(expert_path=(os.getcwd() + '/dataset.npz'), traj_limitation=-1)
        return ExpertDatasetLoader.dataset

class CheckEvalCallback(BaseCallback):
    """
    A custom callback that checks agent's evaluation every predefined number of steps.
    :param model_dir: (str) directory path for model save
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    :param save_interval: (int) Number of timestamps between best mean model saves
    """

    def __init__(self, model_dir, verbose, save_interval=2000):
        super(CheckEvalCallback, self).__init__(verbose)
        self._best_model_path = model_dir
        self._last_model_path = model_dir
        self._best_mean_reward = -np.inf
        self._save_interval = save_interval
        self._best_rew = -1e6

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """


    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # print('_on_rollout_start')

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        rew = self.locals['self'].episode_reward[0]

        # if (self.num_timesteps + 1) % self._save_interval == 0:
        if (rew > self._best_rew):
            # Evaluate policy training performance

            # episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
            #                                                    n_eval_episodes=100,
            #                                                    render=False,
            #                                                    deterministic=True,
            #                                                    return_episode_rewards=True)



            # mean_reward = round(float(np.mean(self.locals['episode_rewards'][-101:-1])), 1)


            # print(self.num_timesteps + 1, 'timesteps')
            # print("Best mean reward: {:.2f} - Last mean reward: {:.2f}".format(self._best_mean_reward, mean_reward))
            print("Best  reward: {:.2f} - Last best reward: {:.2f}".format(self._best_rew, rew))
            # New best model, save the agent
            # if mean_reward > self._best_mean_reward:
            #     self._best_mean_reward = mean_reward
            #     print("Saving new best model")
            #     self.model.save(self._best_model_path + '_rew_' + str(np.round(self._best_mean_reward, 2)))
            self._best_rew = rew
            print("Saving new best model")
            self.model.save(self._best_model_path + '_rew_' + str(np.round(self._best_rew, 2)))
            path = self._last_model_path + '_' + str(time.localtime().tm_mday) + '_' + str(
                time.localtime().tm_hour) + '_' + str(time.localtime().tm_min)
            global BEST_MODELS_NUM
            BEST_MODELS_NUM=BEST_MODELS_NUM+1
            self.model.save(path)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # print('_on_rollout_end')
        # print('locals', self.locals)
        # print('globals', self.globals)
        # print('n_calls', self.n_calls)
        # print('num_timesteps', self.num_timesteps)
        # print('training_env', self.training_env)


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print('_on_training_end')


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional tensor
        if not self.is_tb_set:
            with self.model.graph.as_default():
                tf.summary.scalar('value_target', tf.reduce_mean(self.model.value_target))
                self.model.summary = tf.summary.merge_all()
            self.is_tb_set = True
        # Log scalar value (here a random variable)


        global BEST_MODELS_NUM
        value = BEST_MODELS_NUM

        env = self.locals['self'].env.unwrapped.envs[0]
        summary1 = tf.Summary(value=[tf.Summary.Value(tag='best_models', simple_value=value)])
        summary2 = tf.Summary(value=[tf.Summary.Value(tag='last_rt', simple_value=env.last_rt)])
        summary3 = tf.Summary(value=[tf.Summary.Value(tag='last_final_reward', simple_value=env.last_final_reward)])
        self.locals['writer'].add_summary(summary1, self.num_timesteps)
        self.locals['writer'].add_summary(summary2, self.num_timesteps)
        self.locals['writer'].add_summary(summary3, self.num_timesteps)
        return True

def data_saver(obs, act, rew, dones, ep_rew):
    user = os.getenv("HOME")
    np.save(user+'/git/SmartLoader/saved_ep/obs', obs)
    np.save(user+'/git/SmartLoader/saved_ep/act', act)
    np.save(user+'/git/SmartLoader/saved_ep/rew', rew)

    ep_str = [False] * len(dones)
    ep_str[0] = True

    for i in range(len(dones) - 1):
        if dones[i]:
            ep_str[i + 1] = True

    np.save(user+'/git/SmartLoader/saved_ep/ep_str', ep_str)
    np.save(user+'/git/SmartLoader/saved_ep/ep_ret', ep_rew)


def build_model(algo, policy, env_name, log_dir, expert_dataset=None):
    """
    Initialize model according to algorithm, architecture and hyperparameters
    :param algo: (str) Name of rl algorithm - 'sac', 'ppo2' etc.
    :param env_name:(str)
    :param log_dir:(str)
    :param expert_dataset:(ExpertDataset)
    :return:model: stable_baselines model
    """
    from stable_baselines.common.vec_env import DummyVecEnv
    model = None
    if algo == 'sac':
        # policy_kwargs = dict(layers=[64, 64, 64],layer_norm=False)

        # model = SAC(policy, env_name, gamma=0.99, learning_rate=1e-4, buffer_size=500000,
        #             learning_starts=5000, train_freq=500, batch_size=64, policy_kwargs=policy_kwargs,
        #             tau=0.01, ent_coef='auto_0.1', target_update_interval=1,
        #             gradient_steps=1, target_entropy='auto', action_noise=None,
        #             random_exploration=0.0, verbose=2, tensorboard_log=log_dir,
        #             _init_setup_model=True, full_tensorboard_log=True,
        #             seed=None, n_cpu_tf_sess=None)

        # SAC - start learning from scratch
        # policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[32, 32, 32])
        policy_kwargs = dict(layers=[32, 32, 32], layer_norm=False)

        env = DummyVecEnv([lambda: gym.make(env_name)])
        model = A2C(CnnMlpPolicy, env, verbose=2,gamma=0.99, learning_rate=1e-4,  tensorboard_log=log_dir, _init_setup_model=True, full_tensorboard_log=True,seed=None, n_cpu_tf_sess=None)


        # model = SAC(CnnMlpPolicy, env=env, gamma=0.99, learning_rate=1e-4, buffer_size=50000,
        #             learning_starts=1000, train_freq=1, batch_size=64,
        #             tau=0.01, ent_coef='auto', target_update_interval=1,
        #             gradient_steps=1, target_entropy='auto', action_noise=None,
        #             random_exploration=0.0, verbose=2, tensorboard_log=log_dir,
        #             _init_setup_model=True, full_tensorboard_log=True,
        #             seed=None, n_cpu_tf_sess=None)

    elif algo == 'ppo1':
        model = PPO1('MlpPolicy', env_name, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2,
                     entcoeff=0.01,
                     optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5,
                     schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True,
                     policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)
    elif algo == 'trpo':
        model = TRPO('MlpPolicy', env_name, timesteps_per_batch=4096, tensorboard_log=log_dir, verbose=1)
    elif algo == 'gail':
        assert expert_dataset is not None
        model = GAIL('MlpPolicy', env_name, expert_dataset, tensorboard_log=log_dir, verbose=1)
    assert model is not None
    return model


def pretrain_model(dataset, model):
    # load dataset only once
    # expert_dataset('3_rocks_40_episodes')
    assert (dataset in locals() or dataset in globals()) and dataset is not None
    print('pretrain')
    model.pretrain(dataset, n_epochs=2000)


def record(env):
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


def play(save_dir, env):
    model = SAC.load(save_dir + '/model_dir/sac/test_25_25_14_15', env=env,
                     custom_objects=dict(learning_starts=0))  ### ADD NUM
    for _ in range(2):

        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # print('state: ', obs[0:3], 'action: ', action)


def train(algo, policy, pretrain, n_timesteps, log_dir, model_dir, env_name, model_save_interval):
    """
    Train an agent
    :param algo: (str)
    :param policy: type of network (str)
    :param pretrain: (bool)
    :param n_timesteps: (int)
    :param log_dir: (str)
    :param model_dir: (str)
    :param env_name: (str)
    :return: None
    """
    dataset = ExpertDatasetLoader() if pretrain or algo == 'gail' else None
    model = build_model(algo=algo, policy=policy, env_name=env_name, log_dir=log_dir, expert_dataset=dataset)
    if pretrain:
        pretrain_model(dataset, model)

    # learn
    print("learning model type", type(model))
    custom_eval_callback = CheckEvalCallback(model_dir, verbose=2, save_interval=model_save_interval)
    # eval_callback = EvalCallback(env_name=env_name, best_model_save_path=model_dir,
    #                              log_path='./logs/', eval_freq=model_save_interval,
    #                              deterministic=True, render=False, callback_on_new_best=on_new_best_model_cb)
    # tensorboard_callback = TensorboardCallback(verbose=2)
    model.learn(total_timesteps=n_timesteps, callback=[custom_eval_callback])
    model.save(env_name)





def CreateLogAndModelDirs(args):
    '''
    Create log and model directories according to algorithm, time and incremental index
    :param args:
    :return:
    '''

    #
    dir = args.dir_pref + args.mission
    model_dir = dir + args.model_dir + args.algo
    log_dir = dir + args.tensorboard_log + args.algo
    os.makedirs(model_dir, exist_ok=True)
    # create new folder
    try:
        tests = os.listdir(model_dir)
        indexes = []
        for item in tests:
            indexes.append(int(item.split('_')[1]))
        if not bool(indexes):
            k = 0
        else:
            k = max(indexes) + 1
    except FileNotFoundError:
        os.makedirs(log_dir)
        k = 0
    suffix = '/test_{}'.format(str(k))
    model_dir = os.getcwd() + '/' + model_dir + suffix
    log_dir += suffix
    logger.configure(folder=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    print('log directory created', log_dir)
    return dir, model_dir, log_dir


def main(args):
    # register_policy('CnnMlpPolicy',CnnMlpPolicy)
    env_name = args.mission + '-' + args.env_ver
    env = gym.make(env_name)  # .unwrapped  <= NEEDED?
    print('gym env created', env_name, env)
    save_dir, model_dir, log_dir = CreateLogAndModelDirs(args)

    if args.job == 'train':
        train(args.algo, args.policy, args.pretrain, args.n_timesteps, log_dir, model_dir, env_name, args.save_interval)
    elif args.job == 'record':
        record(env)
    elif args.job == 'play':
        play(save_dir, env)
    elif args.job == 'BC_agent':
        raise NotImplementedError
    else:
        raise NotImplementedError(args.job + ' is not defined')


def add_arguments(parser):
    parser.add_argument('--mission', type=str, default="PushAlgoryxEnv", help="The agents' task")
    parser.add_argument('--env-ver', type=str, default="v0", help="The custom gym enviornment version")
    parser.add_argument('--dir-pref', type=str, default="stable_bl/", help="The log and model dir prefix")

    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='/log_dir/', type=str)
    parser.add_argument('-mdl', '--model-dir', help='model directory', default='/model_dir/', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='sac', type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--policy', help='Network topography', default='CnnMlpPolicy', type=str, required=False, choices=POLICIES)

    parser.add_argument('--job', help='job to be done', default='train', type=str, required=False, choices=JOBS)
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=int(1e6), type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1, type=int)
    parser.add_argument('--save-interval', help='Number of timestamps between model saves', default=2000, type=int)
    parser.add_argument('--eval-freq', help='Evaluate the agent every n steps (if negative, no evaluation)',
                        default=10000, type=int)
    parser.add_argument('--eval-episodes', help='Number of episodes to use for evaluation', default=5, type=int)
    parser.add_argument('--save-freq', help='Save the model every n steps (if negative, no checkpoint)', default=-1,
                        type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1, type=int)
    parser.add_argument('--pretrain', help='Evaluate pretrain phase', default=False, type=bool)
    parser.add_argument('--load-expert-dataset', help='Load Expert Dataset', default=False, type=bool)
    # parser.add_argument('-params', '--hyperparams', type=str, nargs='+', action=StoreDict,
    #                     help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    # parser.add_argument('-uuid', '--uuid', action='store_true', default=False,
    #                     help='Ensure that the run has a unique ID')
    # parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict,
    #                     help='Optional keyword argument to pass to the env constructor')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)