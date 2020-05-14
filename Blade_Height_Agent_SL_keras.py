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


def Supervised_learning(input_states, labels, nn_size, batch_size, lr, epochs, new_model=True, train=True):


    if new_model:   ## create new sequential keras model


        input_size = [input_states.shape[-1]]

        lb_size = 1

        input_layer = Input(shape=input_size, name='xyz_pitch_input')

        fc_l = input_layer
        for i in range(len(nn_size)):
            fc_l = Dense(nn_size[i], activation='relu')(fc_l)

        output = Dense(lb_size, activation='relu', use_bias=True)(fc_l)

        model = Model(inputs=input_layer, outputs=output)

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
            x=input_states,
            y=labels,
            batch_size=batch_size,
            verbose=2,
            epochs=epochs,
            validation_split=0.3
        )
        model.save('/home/graphics/git/SmartLoader/saved_models/lift_est_model_corrected')

    # for _ in range(num_of_evals):
    #
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #
    #     index = np.random.randint(len(heat_maps))
    #     h_map = heat_maps[index, :, :]
    #
    #     est_pos = model.predict(h_map.reshape(1, 1, 260, 60))
    #
    #     shovle_pos = [est_pos[0][0] * 260, est_pos[0][1] * (map_clip[1] - map_clip[0])]
    #
    #     ax.imshow(h_map.squeeze())
    #
    #     plt.scatter(shovle_pos[1], shovle_pos[0], s=300, c='red', marker='o')
    #
    #     plt.show()


###########  labels[550]  #####  model.predict(states[550].reshape([1,ob_size]))
def main():


    expert_path_1 = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/no_clip/all_recs_no_peel/'
    expert_path_2 = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/no_clip/lift_23_ep/'

    # expert_path = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/sand_levels_RHM/y_clip/'
    # states_1 = np.load(expert_path_1+'states.npy', allow_pickle=True)
    # states_2 = np.load(expert_path_2+'states.npy', allow_pickle=True)

    states = np.load(expert_path_2 + 'states.npy', allow_pickle=True)

    x = np.stack(states[:, 1])[:, 0]
    x = (x - 0) / 2.6

    y = np.stack(states[:, 1])[:, 1]
    y = (y - 0) / 1.6

    z = np.stack(states[:, 1])[:, 2]
    z = (z - np.min(z)) / np.ptp(z)

    lift = np.array(states[:, 4])
    lift = (lift - 150) / 100

    pitch = np.array(states[:, 5])
    pitch = (pitch - 50) / 230

    inputs = np.array([x,y,z,pitch])
    inputs = np.transpose(inputs)

    labels = lift

    nn_size = [128, 64, 32]
    batch_size = 64
    learning_rate = 1e-5
    epochs = 500

    Supervised_learning(inputs, labels, nn_size, batch_size, learning_rate, epochs)

if __name__ == '__main__':
    main()