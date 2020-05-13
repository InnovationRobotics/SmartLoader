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


def imitation_learning(buffer, env_id, nn_size, batch_size, lr, epochs, evaluations, new_model = False, train = False):

    env = gym.make(env_id).unwrapped

    ob_size = env.observation_space.shape[0]
    ac_size = env.action_space.shape[0]

    observations = buffer['st']
    observations = np.array(observations)

    labels = buffer['lb']
    labels = np.array(labels)

    if new_model:  ## create new tf graph

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

    else:  ### load existing tf graph

        sess = tf.Session()
        # let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('my_test_model-1000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        X = graph.get_tensor_by_name("X:0")
        Y = graph.get_tensor_by_name("Y:0")

        train_op  = graph.get_tensor_by_name("train_op:0")
        loss = graph.get_tensor_by_name("loss:0")



    # model = SAC.load('/home/graphics/git/SmartLoader/saved_models/test_49_rew_21922.7', env=env,
    #                  custom_objects=dict(learning_starts=0))  ### ADD NUM

    model_path = '/home/graphics/git/SmartLoader/saved_models/' + test_name + '/'

    if train:

        loss_save = []

        for epoch in range(epochs):

            for steps in range(learn_steps):
                batch_index = np.random.choice(len(observations), size=batch_size)  # Batch size

                state_batch, action_batch = observations[batch_index], labels[batch_index]

                _, cur_loss = sess.run([train_op, loss], feed_dict={
                    X: state_batch,
                    Y: action_batch})

                # log_summaries(summary_ops, summary_vars, cur_loss, steps, sess, tbwriter)

                if (steps % 5000) == 0:
                    print('learn steps: ', steps, 'loss: ', cur_loss)
                    print('expert action: ', action_batch[0])
                    print('agent action: ', sess.run( action, feed_dict={ X: state_batch[0].reshape(1, ob_size[-1]) } ) )
                    loss_save.append(cur_loss)
                    checkpoint_path = '/home/graphics/git/SmartLoader/checkpoint_models/' + test_name +'_'+ str(steps) + '/'
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    saver.save(sess, checkpoint_path)

            if not os.path.exists(model_path):
                os.makedirs(model_path)
            saver.save(sess, model_path + 'final_model')

            for episode in range(supervised_episodes):       ### performs demonstrations to correct wrong behaviour

                obs = env.reset()
                done = False
                print('Episode number ', episode+1)

                while not done:

                    act = sess.run( action, feed_dict={ X: state_batch[0].reshape(1, ob_size[-1]) } )
                    new_obs, act, done, info = env.step(act[0])

                    expert_act = model.predict(obs)
                    expert_act = np.hstack( np.array( [expert_act[0], np.array([0])] ))

                    observations = np.concatenate((observations, obs.reshape(1,ob_size[-1])), axis=0)
                    # labels = np.concatenate((labels, expert_act[0].reshape(1,ac_size[-1])), axis=0)
                    labels = np.concatenate((labels, expert_act.reshape(1, ac_size[-1])), axis=0)

                    obs = new_obs

            env.close()


    else:

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

    mission = 'PushStonesEnv'  # Change according to algorithm
    env_id = mission + '-v0'

    expert_path = '/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/'

    states = np.load(expert_path + 'obs.npy')
    labels = np.load(expert_path + 'act.npy')

    replay_buffer = {"st": states, "lb": labels}

    nn_size = [256, 256, 256]
    batch_size = 128
    learning_rate = 1e-4
    epochs = 1000
    evaluations = 50

    imitation_learning(replay_buffer, env_id, nn_size, batch_size, learning_rate, epochs, evaluations)


if __name__ == '__main__':
    main()