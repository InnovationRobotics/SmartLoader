import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Conv1D, Conv2D, Conv3D, MaxPooling2D, Flatten, BatchNormalization, concatenate, Dropout
from keras.optimizers import Adam, SGD, Nadam, Adamax, Adagrad
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model
from matplotlib import pyplot as plt
import numpy as np
import talos
from talos.commands.restore import Restore
import pickle


# translate chosen action (array) to joystick action (dict)


def imitation_learning(heat_maps, actions, x_val, y_val, params):

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


    # print(' ------------ now lets evaluate -------------')

    # loss = []
    # for k in range(len(eval_actions)):
    #     action = model.predict(eval_heat_maps[k].reshape(1,1,100,6))
    #     loss.append(np.abs(action - eval_actions[k]))
    # avg_loss = np.mean(loss)
    # print('avarage loss for {} evaluations: {}'.format(len(eval_actions),avg_loss))
    #



###########  labels[550]  #####  model.predict(states[550].reshape([1,ob_size]))
def main():


    expert_path = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/Push_49_ep/'

    # r_model = Restore('imitation_learning.zip')


    heat_maps = np.load(expert_path+'heatmap.npy')
    states = np.load(expert_path+'states.npy')
    actions = np.load(expert_path + 'actions.npy')

    # p = {'hist_size': [1, 2, 3, 4, 5],
    #      'input_conv_kernel_size': [(2, 2), (3, 3)],
    #      'input_conv_filters': [8, 16, 32],
    #      'conv_dropout': [0.0, 0.1, 0.25],
    #      'conv_layers': [1, 2, 3],
    #      'conv_kernel_size' : [(2, 2), (3, 3)],
    #      'conv_filters' : [8, 16, 32, 64],
    #      'conv_strides' : [(1, 1),(2, 2)],
    #      'batch_norm' : [True, False],
    #      'output_conv_kernel_size' : [(2, 2), (3, 3)],
    #      'output_conv_filters' :[32, 64],
    #      'input_layer_size' : [32, 64, 128],
    #      'hidden_layers' : [1, 2, 3],
    #      'layer_size' : [32, 64, 128],
    #      'output_layer_size': [16, 32, 64, 128],
    #      'bias': [True, False],
    #      'batch_size': [16, 32, 64],
    #      'learning_rate': [1e-4, 5e-4, 1e-5, 5e-5],
    #      'epochs' : [500],
    #      'optimizer' : ['adam', 'nadam', 'sgd'],
    #      'activations' : ['relu', 'elu', 'sigmoid'],
    #      'output_activations' : ['relu', 'elu', 'sigmoid']}

    hist_size = 1
    #
    # heat_map_hist = []
    #
    # for kk in range(len(heat_maps) - hist_size):
    #     heat_map_hist.append(heat_maps[kk:kk + hist_size, :, :])
    #
    # heat_maps = np.array(heat_map_hist)
    # actions = actions[hist_size:]

    heat_maps = heat_maps.reshape(len(heat_maps), hist_size, 100, 6)

    # p = {'input_conv_kernel_size': [(2, 2), (3, 3)],
    #      'input_conv_filters': [8, 16, 32],
    #      'conv_dropout': [0.0, 0.25],
    #      'conv_layers': [1, 2],
    #      'conv_kernel_size': [(2, 2), (3, 3)],
    #      'conv_filters': [8, 16, 32],
    #      'conv_strides': [(1, 1), (2, 2)],
    #      'batch_norm': [False],
    #      'output_conv_kernel_size': [(2, 2), (3, 3), (4, 4)],
    #      'output_conv_filters': [16, 32, 64],
    #      'input_layer_size' : [64, 128, 256],
    #      'hidden_layers': [1, 2, 3],
    #      'layer_size': [64, 128, 256],
    #      'output_layer_size': [16, 32, 64],
    #      'output_bias': [False, True],
    #      'batch_size': [64],
    #      'learning_rate': [1e-4, 2.5e-4, 5e-4],
    #      'epochs': [400],
    #      'optimizer': ['adam'],
    #      'activations': ['relu'],
    #      'output_activations': ['sigmoid']}
    #
    p = {'input_conv_kernel_size': [(2, 2)],
         'input_conv_filters': [8],
         'conv_dropout': [0.0],
         'conv_layers': [1],
         'conv_kernel_size': [(3, 3)],
         'conv_filters': [32],
         'conv_strides': [(1, 1)],
         'batch_norm': [False],
         'output_conv_kernel_size': [(3, 3)],
         'output_conv_filters': [64],
         'input_layer_size': [256],
         'hidden_layers': [1],
         'layer_size': [256],
         'output_layer_size': [64],
         'output_bias': [False],
         'batch_size': [64],
         'learning_rate': [5e-4],
         'epochs': [400],
         'optimizer': ['adam'],
         'activations': ['relu'],
         'output_activations': ['sigmoid']}

    # imitation_learning(heat_maps, actions, params=p)

    t = talos.Scan(x=heat_maps, y=actions, model=imitation_learning, params=p, experiment_name='BC_HM_Push')

    talos.Deploy(scan_object=t, model_name='Push_BC_best_vl',  metric='val_loss', asc=True)
    talos.Deploy(scan_object=t, model_name='Push_BC_best_l', metric='loss', asc=True)



if __name__ == '__main__':
    main()
