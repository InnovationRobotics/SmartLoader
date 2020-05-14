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


def imitation_learning(buffer, env_id, nn_size, batch_size, lr, epochs, new_model=False, train=False):

    env = gym.make(env_id).unwrapped

    ob_size = env.observation_space.shape[0]
    ac_size = env.action_space.shape[0]

    observations = buffer['st']
    observations = np.array(observations)

    labels = buffer['lb']
    labels = np.array(labels)

    if new_model:  ## start new session

        # Observation inputs
        with tf.variable_scope(name_or_scope='Input'):
            X = tf.placeholder(dtype=tf.float32, shape=[None, ob_size])

        # Actions
        with tf.variable_scope(name_or_scope='Output'):
            Y = tf.placeholder(dtype=tf.float32, shape=[None, ac_size])

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
            output = tf.layers.dense(hidden, ac_size, activation=tf.nn.tanh, use_bias=False)

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
        # sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.trainable_variables())

    else:  ### load existing values to graph:

        sess = tf.Session()
        # let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('/home/graphics/git/SmartLoader/saved_models/Push/tf_models/test_model.meta')
        graph = tf.get_default_graph()

        # saver.restore(sess, tf.train.latest_checkpoint('/home/graphics/git/SmartLoader/saved_models/Push/tf_models'))
        saver.restore(sess, '/home/graphics/git/SmartLoader/saved_models/Push/tf_models/test_model')


        # X = graph.get_tensor_by_name('Input/Placeholder:0')
        # action = tf.get_collection('Actions/Placeholder:0')

        # for op in graph.get_operations():
        #     op.name = graph.get_tensor_by_name(op.name+':0')
        # w1 = graph.get_tensor_by_name("w1:0")

    if train:

        losses = []

        model_path = '/home/graphics/git/SmartLoader/saved_models/Push/tf_models/test_model'

        for epoch in range(epochs):

            batch_index = np.random.choice(len(observations), size=batch_size)  # Batch size

            state_batch, action_batch = observations[batch_index], labels[batch_index]

            _, cur_loss = sess.run([train_op, loss], feed_dict={
                X: state_batch,
                Y: action_batch})

            if (epoch % 1000) == 0:
                print('Current Epoch: ', epoch, 'loss: ', cur_loss)
                print('expert action: ', action_batch[0])
                print('agent action: ', sess.run(action, feed_dict={ X: state_batch[0].reshape(1, ob_size) } ) )
                losses.append(cur_loss)

                # saver.save(sess, model_path, global_step=epoch)

        saver.save(sess, model_path)
        # tf.saved_model.simple_save(sess)

    print(' ------------ now lets compare -------------')

    env =  gym.make(env_id).unwrapped

    tests = 20

    for _ in range(tests):
        obs = env.reset()
        done = False
        while not done:
            act = sess.run(action, feed_dict={X: obs.reshape(1, ob_size)})
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
    epochs = int(1e6)

    imitation_learning(replay_buffer, env_id, nn_size, batch_size, learning_rate, epochs)


if __name__ == '__main__':
    main()