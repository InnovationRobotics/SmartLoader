import tensorflow as tf
import numpy as np
import sys
import csv
import gym
import os
import gym_SmartLoader.envs
from stable_baselines import SAC
from stable_baselines import logger

tf.reset_default_graph()


def csv_writer(writer, action, observation):

    saver = []
    for ob in observation:
        saver.append(ob)
    for act in action:
        saver.append(act)

    writer.writerow(saver)


def imitation_learning(buffer, env_id, nn_size, learn_steps, batch_size, lr, epochs, supervised_episodes, learn=False, load_model=False):

    observations = buffer['st']
    observations = np.array(observations)
    ob_size = observations.shape

    labels = buffer['lb']
    labels = np.array(labels)

    # labels = labels.reshape(labels.shape[0], 1)   ### add in case action size is 1

    ac_size = labels.shape

    test_name = 'testy'

    # Observation inputs
    with tf.variable_scope(name_or_scope='Input'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, ob_size[-1]])

    # Actions
    with tf.variable_scope(name_or_scope='Actions_Placeholder'):
        Y = tf.placeholder(dtype=tf.float32, shape=[None, ac_size[-1]])

    # # Hidden Layers:
    for i in range(len(nn_size)):
        scope_name = 'Hidden_{}'.format(i+1)
        if i == 0:
            with tf.variable_scope(name_or_scope=scope_name):
                hidden = tf.layers.dense(X, units=nn_size[i], activation=tf.nn.relu)
        else:
            with tf.variable_scope(name_or_scope=scope_name):
                hidden = tf.layers.dense(hidden, units=nn_size[i], activation=tf.nn.relu)

    with tf.variable_scope(name_or_scope='Output'):
        # Make output layers
        output = tf.layers.dense(hidden, ac_size[-1], activation=tf.nn.tanh, use_bias=False)

    with tf.variable_scope(name_or_scope='Actions'):
        action = output

    with tf.variable_scope(name_or_scope='Loss'):
        loss = tf.reduce_mean(tf.square(Y - output))
        # loss = tf.reduce_mean(tf.losses.mean_squared_error(Y, output))
        tf.summary.scalar('loss', loss)

    with tf.variable_scope(name_or_scope='training'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss=loss)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    # saver = tf.train.Saver()

    if load_model:

        tf.train.import_meta_graph('/home/graphics/git/SmartLoader/saved_models/good_agent/final_model.meta')
        # saver.restore(sess, "/home/graphics/git/SmartLoader/saved_models/good_agent/final_model")

    mission = 'PushStonesEnv'  # Change according to algorithm
    env = gym.make(mission + '-v0').unwrapped
    # model = SAC.load('/home/graphics/git/SmartLoader/saved_models/test_49_rew_21922.7', env=env,
    #                  custom_objects=dict(learning_starts=0))  ### ADD NUM

    model_path = '/home/graphics/git/SmartLoader/saved_models/' + test_name + '/'

    # if learn:
    #
    #     loss_save = []
    #
    #     for epoch in range(epochs):         # num of epochs for backprop learning and dagger demonstrations
    #
    #         for steps in range(learn_steps):
    #             batch_index = np.random.choice(len(observations), size=batch_size)  # Batch size
    #
    #             state_batch, action_batch = observations[batch_index], labels[batch_index]
    #
    #             _, cur_loss = sess.run([train_op, loss], feed_dict={
    #                 X: state_batch,
    #                 Y: action_batch})
    #
    #             # log_summaries(summary_ops, summary_vars, cur_loss, steps, sess, tbwriter)
    #
    #             if (steps % 5000) == 0:
    #                 print('learn steps: ', steps, 'loss: ', cur_loss)
    #                 print('expert action: ', action_batch[0])
    #                 print('agent action: ', sess.run( action, feed_dict={ X: state_batch[0].reshape(1, ob_size[-1]) } ) )
    #                 loss_save.append(cur_loss)
    #                 checkpoint_path = '/home/graphics/git/SmartLoader/checkpoint_models/' + test_name +'_'+ str(steps) + '/'
    #                 if not os.path.exists(checkpoint_path):
    #                     os.makedirs(checkpoint_path)
    #                 saver.save(sess, checkpoint_path)
    #
    #         if not os.path.exists(model_path):
    #             os.makedirs(model_path)
    #         saver.save(sess, model_path + 'final_model')
    #
    #         for episode in range(supervised_episodes):       ### performs demonstrations to correct wrong behaviour
    #
    #             obs = env.reset()
    #             done = False
    #             print('Episode number ', episode+1)
    #
    #             while not done:
    #
    #                 act = sess.run( action, feed_dict={ X: state_batch[0].reshape(1, ob_size[-1]) } )
    #                 new_obs, act, done, info = env.step(act[0])
    #
    #                 expert_act = model.predict(obs)
    #                 expert_act = np.hstack( np.array( [expert_act[0], np.array([0])] ))
    #
    #                 observations = np.concatenate((observations, obs.reshape(1,ob_size[-1])), axis=0)
    #                 # labels = np.concatenate((labels, expert_act[0].reshape(1,ac_size[-1])), axis=0)
    #                 labels = np.concatenate((labels, expert_act.reshape(1, ac_size[-1])), axis=0)
    #
    #                 obs = new_obs
    #
    #         env.close()
    #
    #     np.savetxt(test_name + 'losses.txt', np.array([loss_save]))

    # else:

    tf.train.import_meta_graph('/home/graphics/git/SmartLoader/saved_models/BC_4/final_model.meta')
    # saver.restore(sess, "/home/graphics/git/SmartLoader/saved_models/BC_4/final_model")

    print(' ------------ now lets compare -------------')

    env = gym.make(env_id).unwrapped

    tests = 20

    for _ in range(tests):
        obs = env.reset()
        done = False
        while not done:
            act = sess.run(action, feed_dict={X: obs.reshape(1, ob_size[-1])})
            obs, reward, done, info = env.step(act[0])

    sess.close()
    env.close()

def main():

    # mission = 'PushStonesEnv'  # Change according to algorithm
    # env_id = mission + '-v0'
    # env = gym.make(env_id).unwrapped
    #
    # obs_size = env.observation_space.shape[0]#-3#-10+3
    # act_size = env.action_space.shape[0]+1#-1
    #
    # expert_path = '/home/graphics/git/SmartLoader/saved_experts/1_rock/BC_pusher_straight_50_episodes_extended.csv'
    #
    # # states = np.load(expert_path + 'obs.npy')
    # # labels = np.load(expert_path + 'act.npy')
    #
    #
    # nn_size = [128, 128, 128]
    # learn_steps = int(1e6)
    # batch_size = 64
    # learning_rate = 1e-4
    # epochs = 1
    # supervised_episodes = 0
    #
    # imitation_learning(replay_buffer, env_id, nn_size, learn_steps, batch_size, learning_rate, epochs, supervised_episodes)

    mission = 'PushStonesEnv'  # Change according to algorithm
    env_id = mission + '-v0'
    env = gym.make(env_id).unwrapped

    obs_size = env.observation_space.shape[0]-3-10+3
    act_size = env.action_space.shape[0]-1

    expert_file = '/home/graphics/git/SmartLoader/saved_experts/1_rock/BC_pusher_straight_50_episodes_extended.csv'

    replay_buffer = {"st": [], "lb": []}

    with open(expert_file) as csvfile:
        readCSV = csv.reader(csvfile)
        for row in readCSV:
            arr = []
            for num in row:
                num = float(num)
                arr.append(num)

            replay_buffer["st"].append(np.array(arr[0:obs_size]))
            replay_buffer["lb"].append(arr[obs_size:obs_size+act_size])
            # replay_buffer["st"].append(np.array(arr[0:obs_size]))
            # replay_buffer["lb"].append(arr[obs_size:obs_size+act_size])

    nn_size = [128, 128, 128]
    learn_steps = int(2e5)
    batch_size = 128
    learning_rate = 1e-4
    epochs = 5
    num_evaluations = 5
    learn_episodes = 10
    imitation_learning(replay_buffer, env_id, nn_size, learn_steps, batch_size, learning_rate, epochs, num_evaluations, learn_episodes)


if __name__ == '__main__':
    main()