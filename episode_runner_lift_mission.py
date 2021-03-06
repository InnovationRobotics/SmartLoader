from keras.models import load_model
from SmartLoaderIRL import SmartLoader
from matplotlib import pyplot as plt
from LLC import LLC_pid
import numpy as np
import math
import time
from signal import signal, SIGKILL


def show_map(hmap):
    plt.imshow(hmap)
    plt.show(block=False)
    plt.pause(0.01)
    # plt.show()
    # plt.close


def de_norm_states(des):
    des = [des[0] * 3, des[1] * 130 + 150, des[2] * 230 + 50]
    return des


def normalize_map(h_map):
    norm_hmap = (h_map - np.min(h_map)) / np.ptp(h_map)
    return norm_hmap


def map_clipper(hmap, shovel_pos, x_clip=None, x_offset=None, y_clip=None):

    y_map_clip = 30

    if x_clip:
        x_map_min = int(shovel_pos[0] * 100 + x_offset)
        x_map_max = int(shovel_pos[0] * 100 + x_offset + x_clip)
    else:
        x_map_min = 0
        x_map_max = 260

    if y_clip:
        y_map_min = int(shovel_pos[1] * 100 - y_clip)
        y_map_max = int(shovel_pos[1] * 100 + y_clip)

    end_offset = max(x_map_max-260,0)

    if (x_clip and y_clip):
        clipped_heatmap = np.zeros([x_clip, y_clip*2])
        clipped_heatmap[0:x_clip - end_offset, 0:y_map_clip * 2] = hmap[x_map_min:x_map_max,
                                                                             y_map_min:y_map_max]
    elif (y_clip and not x_clip):
        clipped_heatmap = np.zeros([260, y_clip * 2])
        clipped_heatmap[:, 0:y_map_clip * 2] = hmap[:, y_map_min:y_map_max]
    else:
        clipped_heatmap = hmap

    # only_y_clip = True
    # if only_y_clip:
    #     clipped_heatmap = np.zeros([260, y_map_clip * 2])
    #     clipped_heatmap[:, 0:y_map_clip * 2] = hmap[:, y_map_min:y_map_max]

    return clipped_heatmap


def desired_config(obs, x_model, lift_pitch_model, show_h_map=False):
    # from model predict desired configuration
    blade_pos = [obs['x_blade'], obs['y_blade']]
    hmap = obs['h_map']

    norm_hmap = normalize_map(hmap)

    Lift_Pitch_hmap = map_clipper(norm_hmap, blade_pos, x_clip=100, x_offset=-60, y_clip=30)

    if show_h_map:
        show_map(Lift_Pitch_hmap)

    Thrust_hmap = map_clipper(norm_hmap, blade_pos, x_clip=100, x_offset=-60, y_clip=30)

    Lift_Pitch_hmap = Lift_Pitch_hmap.reshape(1, 1, Lift_Pitch_hmap.shape[0], Lift_Pitch_hmap.shape[1])
    Thrust_hmap = Thrust_hmap.reshape(1, 1, Thrust_hmap.shape[0], Thrust_hmap.shape[1])

    L_P_des = lift_pitch_model.predict(Lift_Pitch_hmap)[0]
    x_deses = x_model.predict(Thrust_hmap)[0] * 3

    lifts = L_P_des[[np.arange(0,10,2)]] * 100 + 150
    pitches = L_P_des[[np.arange(1,10,2)]] * 230 + 50
    lift_des = lifts[-1]-5
    pitch_des = pitches[-1]
    x_des = x_deses[-1]

    # x_err = x_deses - obs['x_blade']
    # min_x_err = np.where(x_err > 0.1)
    # if len(min_x_err[0]) == 0:
    #     x_des = obs['x_blade'] + 0.03
    # else:
    #     x_des = np.min(x_err[min_x_err]) + obs['x_blade']

    return [x_des, obs['y_blade'], lift_des, pitch_des]


def pile_pos(obs):
    h_map = obs['h_map']
    # shovel_pos = [obs['x_blade'], obs['y_blade']]
    blade_x_pos = int(obs['x_blade'] * 100)
    blade_y_pos = int(obs['y_blade'] * 100)
    clipped_map = h_map[blade_x_pos+20:, blade_y_pos-30:blade_y_pos+30]
    # clipped_map = map_clipper(h_map, shovel_pos, x_clip=200, x_offset=20, y_clip=30)
    x_ind = np.unravel_index(np.argmax(clipped_map, axis=None), clipped_map.shape)[0]
    max_x_pos = (x_ind+blade_x_pos+20)/100
    z_max = np.max(clipped_map)

    return max_x_pos, z_max


def quatToEuler(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z


def plot_loc(x, y, x_des, y_des):
    # location plot
    length = len(x)
    t = np.linspace(0, length, length)
    fig, (ax_x, ax_y) = plt.subplots(2)
    ax_x.set_title('x pos')
    ax_y.set_title('y pos')

    ax_x.plot(t, x, color='blue')
    ax_y.plot(t, y, color='blue')
    ax_x.plot(t, x_des, color='red')
    ax_y.plot(t, y_des, color='red')

    # save
    fig.savefig('/home/sload/git/SmartLoader/LLC/plots/locations')
    print('figure saved!')


if __name__ == '__main__':

    env = SmartLoader()
    obs = env.get_obs()

    x_model = load_model('/home/sload/Downloads/lift_task_x_model_5_pred_b_wind')
    lift_pitch_model = load_model('/home/sload/Downloads/lift_task_LP_model_5_pred_b_wind')

    X, Y, X_des, Y_des = [], [], [], []
    steps = 0

    # while True:
    #     obs = env.get_obs()
    #     des = desired_config(obs, x_model, lift_pitch_model)
    #     print('curr lift = ', obs['lift'], 'des lift = ', des[2], 'curr pitch = ', obs['pitch'], 'des pitch = ', des[3])
    #     # print('curr pitch = ', obs['pitch'], 'des pitch = ', des[3])
    #     # print('curr x = ', obs['x_blade'], 'des x = ', des[0])

    for step in range(3):

        # pile peak x location and z height
        x_pile, z_pile = pile_pos(obs)

        ##### load mission #####
        load_pid = LLC_pid.LoadPid()
        counter = 0
        last_loc = obs['x_vehicle']

        while True:
            des = desired_config(obs, x_model, lift_pitch_model)
            # print('curr lift = ', obs['lift'], 'des lift = ', des[2])
            action = load_pid.step(obs, des, x_pile)

            # # if lift demand high, son't move' only lift
            # if (des[2] - obs['lift']) > 7:
            #     action[1] = 0

            obs = env.step(action)

            # save locations
            X.append(obs['x_blade'])
            Y.append(obs['y_blade'])
            X_des.append(des[0])
            Y_des.append(des[1])
            steps += 1

            # # stopping conditions
            # movement = abs(last_loc - obs['x_vehicle'])
            # # print(movement)
            # if movement < 0.003:
            #     counter += 1
            #     if counter == 100 and obs['lift'] >= 180:
            #         break
            # else:
            #     counter = 0
            #     last_loc = obs['x_vehicle']

            print('pitch des = ', des[3], 'cur = ', obs['pitch'], 'lift des = ', des[2], 'cur = ', obs['lift'])
            # print('x pile = ', x_pile, 'x cur = ', obs['x_blade'], 'lift des = ', des[2])
            if obs['x_blade'] >= x_pile - 0.1 and obs['z_blade'] - z_pile >= 0.35:  # obs['lift'] >= 190:
                break

        print('Loading mission done!')
        # plots
        load_pid.lift_pid.save_plot('lift load {}'.format(str(step)), 'lift')
        load_pid.pitch_pid.save_plot('pitch load {}'.format(str(step)), 'pitch')
        load_pid.speed_pid.save_plot('speed load {}'.format(str(step)), 'speed')

        ##### dump mission #####
        des = [obs['x_blade'], obs['y_blade'], 200, 220]
        dump_pid = LLC_pid.DumpPid(des)
        while True:
            action = dump_pid.step(obs)
            obs = env.step(action)

            # save locations
            X.append(obs['x_blade'])
            Y.append(obs['y_blade'])
            X_des.append(des[0])
            Y_des.append(des[1])

            # stopping condition
            if obs['lift'] >= des[2] and obs['pitch'] >= des[3]:
                break

        print('dumping mission done!')
        # plots
        dump_pid.lift_pid.save_plot('lift dump {}'.format(str(step)), 'lift')
        dump_pid.pitch_pid.save_plot('pitch dump {}'.format(str(step)), 'pitch')

        ##### drive backwards #####
        des = [obs['x_vehicle']-0.5, obs['y_vehicle'], 180, 140]
        back_pid = LLC_pid.DriveBackPid()
        back_and_lower_pid = LLC_pid.DriveBackAndLowerBladePid(des)

        while obs['x_vehicle'] > des[0] + 0.25:
            action = back_pid.step(obs, des)
            obs = env.step(action)

            # save locations
            X.append(obs['x_blade'])
            Y.append(obs['y_blade'])
            X_des.append(des[0])
            Y_des.append(des[1])

        # plots
        back_pid.speed_pid.save_plot('speed back {}'.format(str(step)), 'speed')

        while True:
            action = back_and_lower_pid.step(obs, des)
            obs = env.step(action)

            # save locations
            X.append(obs['x_blade'])
            Y.append(obs['y_blade'])
            X_des.append(des[0])
            Y_des.append(des[1])

            # stopping condition
            if obs['x_vehicle'] <= des[0] and obs['lift'] <= des[2] and obs['pitch'] <= des[3]:
                print('driving backwards mission done!')
                break

        # plots
        back_and_lower_pid.speed_pid.save_plot('speed back {}'.format(str(step)), 'speed')
        back_and_lower_pid.pitch_pid.save_plot('pitch back {}'.format(str(step)), 'pitch')
        back_and_lower_pid.lift_pid.save_plot('lift back {}'.format(str(step)), 'lift')

    # stop moving
    action = np.array([0, 0, 0, 0])
    obs = env.step(action)

    plot_loc(X, Y, X_des, Y_des)

    # signal(SIGKILL, plot_loc(X, Y, X_des, Y_des))
