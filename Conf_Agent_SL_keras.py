import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Conv1D, Conv2D, Conv3D, MaxPooling2D, Flatten, BatchNormalization, concatenate, Dropout, Activation
from keras.optimizers import Adam, SGD, Nadam, Adamax, Adagrad
from keras.models import Model, load_model, model_from_json
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model
from matplotlib import pyplot as plt
import numpy as np
import talos


# translate chosen action (array) to joystick action (dict)


def Supervised_learning(heat_maps, labels, nn_size, batch_size, lr, epochs, new_model=False, train=False):


    if new_model:   ## create new sequential keras model

        heat_maps = heat_maps.reshape(len(heat_maps), 1, heat_maps.shape[1], heat_maps.shape[2])


        hmap_size = heat_maps.shape[1:]

        lb_size = labels[1].shape[0]

        hmap_input = Input(shape=hmap_size, name='Heat_map_input')

        conv_l = hmap_input

        conv_l = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same')(conv_l)
        # conv_l = Activation('elu')(conv_l)
        # conv_l = BatchNormalization()(conv_l)
        conv_l = Dropout(rate=0.25)(conv_l)
        conv_l = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same')(conv_l)
        # conv_l = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_l)

        # conv_l = Conv2D(filters=64, kernel_size=(4, 4), strides=(1, 1), padding='same')(conv_l)
        # conv_l = Dropout(rate=0.25)(conv_l)
        # conv_l = Activation('elu')(conv_l)
        # conv_l = BatchNormalization()(conv_l)
        # conv_l = Dropout(rate=0.1)(conv_l)
        # conv_l = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_l)
        # conv_l = BatchNormalization()(conv_l)
        # conv_l = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_l)
        # conv_l = BatchNormalization()(conv_l)

        conv_l = Flatten()(conv_l)

        # fc_l = concatenate([conv_l, sens_input])
        fc_l = conv_l
        for i in range(len(nn_size)):
            fc_l = Dense(nn_size[i], activation='relu')(fc_l)

        output = Dense(lb_size, activation='relu', use_bias=False)(fc_l)

        model = Model(inputs=hmap_input, outputs=output)

        opt = Adam(lr=lr)
        # opt = Nadam(lr=lr)
        # opt = SGD(lr=lr)

        model.compile(
            loss='mse',
            optimizer=opt)

        print(model.summary())

        # test_name = 'keras_test_1'
        # log_dir = '/home/graphics/git/SmartLoader/log_dir/' + test_name + '/'
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    else:   ### load existing sequential model

        model = load_model('/home/graphics/git/SmartLoader/saved_models/RHM_Push_49_ep_8e-4_loss')

    if train:   ## train new agent
        hist = model.fit(
            x=heat_maps,
            y=labels,
            batch_size=batch_size,
            verbose=2,
            epochs=epochs,
            validation_split=0.3
        )
        model.save('/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/Push_49_ep/full_map/KERAS_conf_est_model')

    num_of_evals = 10
    map_clip = [30, 90]
    for _ in range(num_of_evals):

        fig, ax = plt.subplots(figsize=(10, 10))

        index = np.random.randint(len(heat_maps))
        h_map = heat_maps[index, :, :]

        est_pos = model.predict(h_map.reshape(1, 1, 260, 60))

        shovle_pos = [est_pos[0][0] * 260, est_pos[0][1] * (map_clip[1] - map_clip[0])]

        ax.imshow(h_map.squeeze())

        plt.scatter(shovle_pos[1], shovle_pos[0], s=300, c='red', marker='o')

        plt.show()


###########  labels[550]  #####  model.predict(states[550].reshape([1,ob_size]))
def main():

    # expert_path = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/Push_49_ep/full_map/'
    expert_path = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/Push_49_ep_RHM/norm_and_clip/'

    heat_maps = np.load(expert_path+'heatmaps.npy')
    states = np.load(expert_path+'states.npy', allow_pickle=True)
    actions = np.load(expert_path+'actions.npy')
    ep_starts = np.load(expert_path+'starts.npy')
    ep_starts[0] = True

    labels = []
    aug_heat_maps = []

    num_of_step_labels = 4
    label_horizon = 5

    for k in range(label_horizon, len(states)):
        # if ((ep_starts[k]) or (ep_starts[k-1]) or (ep_starts[k-2])):
        #     print('haha')
        if ep_starts[(k - label_horizon+1):(k + 1)].any():
            continue
        # aug_pos = np.array(positions[k]).flatten()
        # aug_states = np.hstack(np.array(states[k]))
        state_pred = []
        for j in range(k-num_of_step_labels, k+1):
            aug_states = np.hstack(states[j][[1, 4, 5]])
            aug_states = np.delete(aug_states, 2)
            state_pred.append(aug_states)

        labels.append(np.hstack(state_pred))
        aug_heat_maps.append(heat_maps[k-label_horizon])

    aug_heat_maps = np.array(aug_heat_maps)
    labels = np.array(labels)

    nn_size = [128, 32]
    batch_size = 32
    learning_rate = 5e-5
    epochs = 500

    ##################### data test #########################
    test_data = False
    if test_data:
        map_clip = [30, 90]
        for _ in range(50):

            fig, ax = plt.subplots(figsize=(10, 10))

            index = np.random.randint(len(aug_heat_maps))

            shovle_pos = [labels[index, 0] * 260, labels[index, 1] * (map_clip[1] - map_clip[0])]
            ax.imshow(aug_heat_maps[index, :, :])

            plt.scatter(shovle_pos[1], shovle_pos[0], s=300, c='red', marker='o')

            plt.show()
    ##################### data test #########################
    labels = actions
    aug_heat_maps = heat_maps
    Supervised_learning(aug_heat_maps, labels, nn_size, batch_size, learning_rate, epochs)

if __name__ == '__main__':
    main()