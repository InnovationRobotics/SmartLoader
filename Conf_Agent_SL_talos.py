import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Conv1D, Conv2D, Conv3D, MaxPooling2D, Flatten, BatchNormalization, concatenate, Dropout
from keras.optimizers import Adam, SGD, Nadam, Adamax, Adagrad
from keras.models import Model, load_model, model_from_json
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model
from matplotlib import pyplot as plt
from matplotlib.pyplot import gca
import numpy as np
import talos
import zipfile
from talos import Restore
import pickle


# translate chosen action (array) to joystick action (dict)


def supervised_learning(heat_maps, actions, x_val, y_val, params):

    hmap_size = heat_maps.shape[1:]
    ac_size = actions[0].shape[0]

    hmap_input = Input(shape=hmap_size, name='Heat_map_input')

    conv_l = hmap_input
    conv_l = Conv2D(filters=params['input_conv_filters'], kernel_size=params['input_conv_kernel_size'], strides=(1, 1), padding='same', activation='relu')(conv_l)
    conv_l = Dropout(rate=params['conv_dropout'])(conv_l)

    for _ in range(params['conv_layers']):
        conv_l = Conv2D(filters=params['conv_filters'], kernel_size=params['conv_kernel_size'], strides=params['conv_strides'], padding='same', activation='relu')(conv_l)
        if params['batch_norm']:
            conv_l = BatchNormalization()(conv_l)

    conv_l = Conv2D(filters=params['output_conv_filters'], kernel_size=params['output_conv_kernel_size'], strides=(1, 1), padding='same', activation='relu')(conv_l)
    conv_l = Dropout(rate=params['conv_dropout'])(conv_l)

    conv_l = Flatten()(conv_l)

    # fc_l = concatenate([conv_l, sens_input])

    fc_l = conv_l
    fc_l = Dense(params['input_layer_size'], activation=params['activations'])(fc_l)

    for _ in range(params['hidden_layers']):
        fc_l = Dense(params['layer_size'], activation=params['activations'])(fc_l)

    fc_l = Dense(params['output_layer_size'], activation=params['activations'])(fc_l)

    output = Dense(ac_size, activation=params['output_activations'], use_bias=params['output_bias'])(fc_l)

    model = Model(inputs=hmap_input, outputs=output)

    if params['optimizer'] == 'adam':
        opt = Adam(lr=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        opt = SGD(lr=params['learning_rate'])
    elif params['optimizer'] == 'nadam':
        opt = Nadam(lr=params['learning_rate'])
    elif params['optimizer'] == 'adamax':
        opt = Adamax(lr=params['learning_rate'])
    elif params['optimizer'] == 'nadam':
        opt = Nadam(lr=params['learning_rate'])

    model.compile(
        loss='mean_squared_error',
        optimizer=opt)

    print(model.summary())

    hist = model.fit(
        x=heat_maps,
        y=actions,
        batch_size=params['batch_size'],
        #callbacks=[talos.utils.live()],
        verbose=2,
        epochs=params['epochs'],
        validation_data=[x_val, y_val])


    return hist, model

    # x = range(0, params['epochs'])
    # plt.plot(x, hist.history['loss'], label="Training Loss")
    # plt.plot(x, hist.history['val_loss'], label="Eval Loss")
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss Value')
    # plt.legend(loc="upper left")
    # plt.grid(True)
    # plt.show()

def lift_normalize_states(shovle_pos, arm_pitch):
    global y_map_clip
    shovle_pos = [shovle_pos[0] / 2.6, (shovle_pos[1]) / 1.6, (shovle_pos[2]-0.097)/(0.323-0.097)]
    arm_pitch = (arm_pitch-50) / 230  # full span = [50:280]

    return shovle_pos, arm_pitch

def map_normalize_states(shovle_pos, arm_pitch):
    global y_map_clip
    shovle_pos = [shovle_pos[0] / 2.6, (shovle_pos[1] - y_map_clip/100) / (y_map_clip/50), shovle_pos[2]]
    arm_pitch = (arm_pitch-50) / 230  # full span = [50:280]

    return shovle_pos, arm_pitch

def map_clipper(heat_map, shovle_pos, x_map_front_clip, x_map_front_offset, y_map_clip):

    if x_map_front_clip:
        x_map_min = int(shovle_pos[0] * 100 + x_map_front_offset)
        x_map_max = int(shovle_pos[0] * 100 + x_map_front_offset + x_map_front_clip)

        end_offset = max(x_map_max - 260, 0)
        if end_offset > ((x_map_front_clip + x_map_front_offset) * 2 / 3):
            return None

        clipped_heatmap = np.ones([x_map_front_clip, 160])*0.1
        clipped_heatmap[0:x_map_front_clip - end_offset, :] = heat_map[x_map_min:x_map_max, :]
        heat_map = clipped_heatmap

    if y_map_clip:
        y_map_min = int(shovle_pos[1] * 100 - y_map_clip)
        y_map_max = int(shovle_pos[1] * 100 + y_map_clip)

        heat_map = heat_map[:, y_map_min:y_map_max]

    return heat_map

def show_heatmap(heatmap, arm_height, anim=True):
    if anim:
        plt.imshow(heatmap, aspect=1)
        plt.clim(-0.5, 0.5)
        plt.text(00, 000, int(arm_height), fontsize=15)
        plt.show(block=False)
        plt.pause(0.01)
        try:
            del gca().texts[-1]
        except:
            pass
    else:
        plt.imshow(heatmap, aspect=1)
        plt.text(20, 100, int(arm_height), fontsize=15)
        plt.show()
        try:
            del gca().texts[-1]
        except:
            pass

def main():
    global y_map_clip
    expert_path = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/lift_ep_12_5/'
    lift_est_model = load_model('/home/graphics/git/SmartLoader/saved_models/lift_est_model_corrected')

    heat_maps = np.load(expert_path+'heatmaps.npy')
    states = np.load(expert_path+'states.npy', allow_pickle=True)
    actions = np.load(expert_path+'actions.npy')
    ep_starts = np.load(expert_path+'starts.npy')
    ep_starts[0] = True

    # rand_ind = np.random.randint(len(states), size=100)
    # fictive_maps = np.array([np.ones(heat_maps[0].shape)*0.1]*100)
    # fictive_states = states[rand_ind]
    # fictive_states[:, 5] = 165
    # fictive_states[:, 4] = 140
    # fictive_ep_starts = ep_starts[rand_ind]
    # heat_maps = np.vstack([heat_maps, fictive_maps])
    # states = np.vstack([states, fictive_states])
    # ep_starts = np.hstack([ep_starts, fictive_ep_starts])

    labels = []
    aug_heat_maps = []

########################################
    # organize data to fit mission:
########################################

    label_horizon = 5  # how far to the future will the network predict
    num_of_step_labels = 5  # how many future steps will the network predict

    x_map_front_clip = 100  # window size in front of blade
    x_map_front_offset = -60  # offset size in front of blade
    y_map_clip = 30  # window size on sides of the blade

    learn = False
    test_name = 'lift_task_LP_model_5_pred_b_wind'
    if learn:

        for k in range(label_horizon+300, len(states)):

            state_pred = []

            if ep_starts[(k - label_horizon+1):(k + 1)].any():
                continue

            # aug_map = map_clipper(heat_maps[k], states[k, 1], x_map_front_clip,
            #                       x_map_front_offset, y_map_clip)
            # t_lift_shovle_pos, t_lift_pitch_state = lift_normalize_states(states[k][1], states[k][5])
            # t_lift_state = lift_est_model.predict(np.hstack([t_lift_shovle_pos, t_lift_pitch_state]).reshape(1, 4))
            # show_heatmap(aug_map, t_lift_state)

            for j in range(k-num_of_step_labels+1, k+1):

                shovle_pos = states[j][1]
                pitch_state = states[j][[5]]  # only pitch values
                lift_shovle_pos, lift_pitch_state = lift_normalize_states(shovle_pos, pitch_state)

                lift_state = lift_est_model.predict(np.hstack([lift_shovle_pos, lift_pitch_state]).reshape(1, 4))

                shovle_pos, pitch_state = map_normalize_states(shovle_pos, pitch_state)

                # aug_states = [lift_state, pitch_state]  ## for LP model

                aug_states = shovle_pos[0]  ## for x model

                # aug_states[0:3] -= states[k - label_horizon][1]  ## for moving set-point

                # aug_states = actions[j,1]  ## for thrust model

                state_pred.append(aug_states)



            aug_map = map_clipper(heat_maps[k-label_horizon], states[k-label_horizon, 1], x_map_front_clip, x_map_front_offset, y_map_clip)

            state_pred = np.array(state_pred).squeeze()

            # show_heatmap(heat_maps[k - label_horizon], state_pred[-1][0] * 100 + 150)
            # show_heatmap(aug_map, state_pred[-1][0]*100+150)

            if not aug_map.any():
                continue

            labels.append(np.array(state_pred).reshape(np.prod(state_pred.shape)))
            aug_heat_maps.append(aug_map)

            # if (k % 20) == 0:
            #     plt.close()
            # if ep_starts[k+label_horizon]:
            #     plt.close()

        aug_heat_maps = np.array(aug_heat_maps)
        aug_heat_maps = aug_heat_maps.reshape([aug_heat_maps.shape[0], 1, aug_heat_maps.shape[1], aug_heat_maps.shape[2]])
        labels = np.array(labels)

        p = {'input_conv_kernel_size': [(3, 3)],
             'input_conv_filters': [16],
             'conv_dropout': [0.25],
             'conv_layers': [2],
             'conv_kernel_size': [(2, 2)],
             'conv_filters': [16],
             'conv_strides': [(1, 1)],
             'batch_norm': [False],
             'output_conv_kernel_size': [(3, 3)],
             'output_conv_filters': [16],
             'input_layer_size': [256],
             'hidden_layers': [1],
             'layer_size': [256],
             'output_layer_size': [32],
             'output_bias': [False],
             'batch_size': [64],
             'learning_rate': [5e-5],
             'epochs': [200],
             'optimizer': ['adam'],
             'activations': ['relu'],
             'output_activations': ['sigmoid']}
        fract_lim = 1


        # labels = actions
        # aug_heat_maps = heat_maps.reshape(heat_maps.shape[0],1,heat_maps.shape[1],heat_maps.shape[2])
        t = talos.Scan(x=aug_heat_maps, y=labels, model=supervised_learning, params=p, fraction_limit=fract_lim, experiment_name=test_name)

        talos.Deploy(scan_object=t, model_name=test_name+'_best_vl',  metric='val_loss', asc=True)

    else:

        json_file = open('/home/graphics/git/SmartLoader/'+test_name+'_best_vl/'+test_name+'_best_vl_model.json', 'rb')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights('/home/graphics/git/SmartLoader/'+test_name+'_best_vl/'+test_name+'_best_vl_model.h5')

        print(' ------------ now lets evaluate -------------')

        evaluate = False
        if evaluate:
            loss = []
            evals = 50
            for k in range(evals):
                index = np.random.randint(len(aug_heat_maps))
                states = model.predict(aug_heat_maps[index, :, :, :].reshape(1, 1, aug_heat_maps.shape[2], aug_heat_maps.shape[3]))
                arm_height = states[0][0]*50+150
                plt.imshow(aug_heat_maps[index, :, :, :].squeeze(), aspect=1)
                plt.text(40, 5, arm_height, fontsize=15)
                print(labels[index][0] * 50 + 150)
                plt.show()

        model.save('/home/graphics/git/SmartLoader/saved_models/'+test_name)

        print('saved')

if __name__ == '__main__':
    main()
