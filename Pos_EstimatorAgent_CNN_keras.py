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
from os import walk


# translate chosen action (array) to joystick action (dict)


def Supervised_learning(heat_maps, labels, nn_size, batch_size, lr, epochs, new_model=False, train=True):


    if new_model:   ## create new sequential keras model

        heat_maps = heat_maps.reshape(len(heat_maps), 1, 100, 16)

        lb_size = labels.shape[0]

        labels = labels.transpose()

        hmap_size = heat_maps.shape[1:]

        hmap_input = Input(shape=hmap_size, name='Heat_map_input')

        conv_l = hmap_input

        conv_l = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same')(conv_l)
        # conv_l = Activation('elu')(conv_l)
        # conv_l = BatchNormalization()(conv_l)
        # conv_l = Dropout(rate=0.25)(conv_l)
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

        output = Dense(lb_size, activation='relu', use_bias=True)(fc_l)

        model = Model(inputs=hmap_input, outputs=output)

        opt = Adam(lr=lr)
        # opt = Nadam(lr=lr)
        # opt = SGD(lr=lr)

        model.compile(
            loss='mean_absolute_error',
            optimizer=opt)

        print(model.summary())

        # test_name = 'keras_test_1'
        # log_dir = '/home/graphics/git/SmartLoader/log_dir/' + test_name + '/'
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    else:   ### load existing sequential model

        model = load_model('/home/sload/git/SmartLoader/saved_experts/pos_est_model_valsplit0.05')

        heat_maps = heat_maps.reshape(len(heat_maps), 1, 100, 16)

        labels = labels.transpose()


    if train:   ## train new agent
        hist = model.fit(
            x=heat_maps,
            y=labels,
            batch_size=batch_size,
            verbose=2,
            epochs=epochs,
            validation_split=0.05
        )
        model.save('/home/sload/git/SmartLoader/saved_experts/pos_est_model_valsplit0.05_2')

    else:

        num_of_evals = 10

        for _ in range(num_of_evals):
            fig, ax = plt.subplots(figsize=(10, 10))
            index = np.random.randint(len(heat_maps))
            h_map = heat_maps[index, :, :]

            est_pos = model.predict(h_map.reshape(1, 1, 100, 7))

            ax.imshow(h_map)

            plt.scatter(est_pos[0][0], est_pos[0][1], s=100, c='red', marker='o')
            plt.scatter(est_pos[0][2], est_pos[0][3], s=100, c='red', marker='o')

            plt.show()

            # print(labels[index])


###########  labels[550]  #####  model.predict(states[550].reshape([1,ob_size]))
def main():

    expert_path = '/home/sload/xy_locations_03_05/'

    heatmap = np.load(expert_path + 'heatmap.npy', allow_pickle=True)
    labels = np.load(expert_path + 'labels.npy', allow_pickle=True)

    nn_size = [128, 32]
    batch_size = 64
    learning_rate = 4e-5
    epochs = 2000

    Supervised_learning(heatmap, labels, nn_size, batch_size, learning_rate, epochs)

if __name__ == '__main__':
    main()