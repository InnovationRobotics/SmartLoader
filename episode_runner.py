from keras.models import load_model
from SmartLoaderIRL import SmartLoader
from matplotlib import pyplot as plt
from LLC import LLC_pid
import numpy as np
import math
import time


def show_map(hmap):
    plt.imshow(hmap)
    plt.show(block=False)
    plt.pause(0.01)
    # plt.show()
    # plt.close

def de_norm_states(des):
    des = [des[0] * 3, des[1] * 130 + 150, des[2] * 230 + 50]  # un-normalize
    return des

def normalize_map(h_map):
    norm_hmap = (h_map - np.min(h_map)) / np.ptp(h_map)
    return norm_hmap

def map_clipper(hmap, shovle_pos, x_clip=None, x_offset=None, y_clip=None):

    y_map_clip = 30

    if x_clip:
        x_map_min = int(shovle_pos[0] * 100 + x_offset)
        x_map_max = int(shovle_pos[0] * 100 + x_offset + x_clip)
    else:
        x_map_min = 0
        x_map_max = 260

    if y_clip:
        y_map_min = int(shovle_pos[1] * 100 - y_clip)
        y_map_max = int(shovle_pos[1] * 100 + y_clip)

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

def desired_config(obs, x_model, lift_pitch_model):
    # from model predict desired configuration
    blade_pos = [obs['x_blade'], obs['y_blade']]
    hmap = obs['h_map']

    norm_hmap = normalize_map(hmap)

    Lift_Pitch_hmap = map_clipper(norm_hmap, blade_pos, x_clip=50, x_offset=20, y_clip=30)
    Thrust_hmap = map_clipper(norm_hmap, blade_pos, y_clip=30)

    Lift_Pitch_hmap = Lift_Pitch_hmap.reshape(1, 1, Lift_Pitch_hmap.shape[0], Lift_Pitch_hmap.shape[1])
    Thrust_hmap = Thrust_hmap.reshape(1, 1, Thrust_hmap.shape[0], Thrust_hmap.shape[1])

    L_P_des = lift_pitch_model.predict(Lift_Pitch_hmap)[0]
    x_deses = x_model.predict(Thrust_hmap)[0] * 3

    lifts = np.array([L_P_des[0], L_P_des[2], L_P_des[4], L_P_des[6], L_P_des[8]]) * 50 + 150
    pitches = np.array([L_P_des[1], L_P_des[3], L_P_des[5], L_P_des[7], L_P_des[9]]) * 230 + 50

    x_err = x_deses - obs['x_blade']
    min_x_err = np.where(x_err > 0.1)
    if len(min_x_err[0]) == 0:
        x_des = obs['x_blade'] + 0.05
    else:
        x_des = np.min(x_err[min_x_err]) + obs['x_blade']

    return [x_des, obs['y_blade'], lifts[0], pitches[0]]

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


if __name__ == '__main__':

    env = SmartLoader()
    obs = env.get_obs()

    x_model = load_model('/home/sload/Downloads/new_test_all_recordings_x_model_10_pred')
    lift_pitch_model = load_model('/home/sload/Downloads/new_test_new_recordings_LP_model')

    X, Y, X_des, Y_des = [], [], [], []
    steps = 0

    for step in range(3):

        ##### push mission #####
        push_pid = LLC_pid.PushPid()
        counter = 0
        last_loc = obs['x_vehicle']

        while True:
            des = desired_config(obs, x_model, lift_pitch_model)
            action = push_pid.step(obs, des)
            obs = env.step(action)

            # save locations
            X.append(obs['x_blade'])
            Y.append(obs['y_blade'])
            X_des.append(des[0])
            Y_des.append(des[1])
            steps += 1

            # stopping conditions
            movement = abs(last_loc - obs['x_vehicle'])
            print(movement)
            if movement < 0.003:
                counter += 1
                if counter == 100:
                    print('got stuck! move back')
                    break
            else:
                counter = 0
                last_loc = obs['x_vehicle']

            if obs['x_blade'] >= 2.2:
                break

        print('pushing mission done!')
        # plots
        push_pid.lift_pid.save_plot('lift push {}'.format(str(step)), 'lift')
        push_pid.pitch_pid.save_plot('pitch push {}'.format(str(step)), 'pitch')
        # push_pid.steer_pid.save_plot('steer push {}'.format(str(step)), 'steer')
        push_pid.speed_pid.save_plot('speed push {}'.format(str(step)), 'speed')


        ##### dump mission #####
        dump_pid = LLC_pid.DumpPid()
        while True:
            action = dump_pid.step(obs)
            obs = env.step(action)

            # save locations
            X.append(obs['x_blade'])
            Y.append(obs['y_blade'])
            X_des.append(des[0])
            Y_des.append(des[1])

            # stopping condition
            if obs['pitch'] >= dump_pid.pitch_pid.SetPoint:
                break

        print('dumping mission done!')
        # plots
        dump_pid.lift_pid.save_plot('lift dump {}'.format(str(step)), 'lift')
        dump_pid.pitch_pid.save_plot('pitch dump {}'.format(str(step)), 'pitch')


        ##### drive backwards #####
        back_pid = LLC_pid.DriveBackPid()
        des = [0.5, obs['y_vehicle'], 155, ]
        # back_pid.steer_pid.SetPoint = 0
        while True:
            action = back_pid.step(obs, des)
            obs = env.step(action)

            # save locations
            X.append(obs['x_blade'])
            Y.append(obs['y_blade'])
            X_des.append(des[0])
            Y_des.append(des[1])

            # stopping condition
            if obs['x_vehicle'] <= 0.5:
                print('driving backwards mission done!')
                break

        # plots
        # back_pid.steer_pid.save_plot('steer back {}'.format(str(step)), 'steer')
        back_pid.speed_pid.save_plot('speed back {}'.format(str(step)), 'speed')

    # stop moving
    action = np.array([0, 0, 0, 0])
    obs = env.step(action)

    # location plot
    length = len(X)
    t = np.linspace(0, length, length)
    fig, (ax_x, ax_y) = plt.subplots(2)
    ax_x.set_title('x pos')
    ax_y.set_title('y pos')

    ax_x.plot(t, X, color='blue')
    ax_y.plot(t, Y, color='blue')
    ax_x.plot(t, X_des, color='red')
    ax_y.plot(t, Y_des, color='red')

    # save
    fig.savefig('/home/sload/git/SmartLoader/LLC/plots/locations')
    print('figure saved!')
