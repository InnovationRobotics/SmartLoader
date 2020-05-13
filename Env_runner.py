from keras.models import load_model
from SmartLoaderIRL import SmartLoader
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from LLC import LLC_pid
import numpy as np
import math
import time
from scipy.spatial.transform import Rotation as R


y_map_clip = 30
x_map_front_offset = 100
x_map_front_clip = 70

def timing_test():

    for _ in range(10):
        start_x_obs = env.get_obs()['x_blade']
        action = np.array([0, 1, 0, 0])
        time.sleep(1)
        x_obs = env.step(action)['x_blade']
        x_obs_err = x_obs - start_x_obs
        print('GOoooooo')
        start_time = time.time()
        while x_obs_err < 0.003:
            x_obs_err = x_obs - start_x_obs
            # print(lift_obs_err)
            x_obs = env.get_obs()['x_blade']

        end_time = time.time()-start_time
        print(end_time)

        action = np.array([0, 0, 0, 0])
        env.step(action)
        time.sleep(1)
    quit()

    ### timing test
    # for _ in range(10):
    #     start_lift_obs = env.get_obs()['lift']
    #     action = np.array([0, 0, 0, 1])
    #     time.sleep(1)
    #     lift_obs = env.step(action)['lift']
    #     lift_obs_err = lift_obs - start_lift_obs
    #     print('GOoooooo')
    #     start_time = time.time()
    #     while lift_obs_err < 4:
    #         lift_obs_err = lift_obs - start_lift_obs
    #         # print(lift_obs_err)
    #         lift_obs = env.get_obs()['lift']
    #
    #     end_time = time.time()-start_time
    #     print(end_time)
    #
    #     action = np.array([0, 0, 0, 0])
    #     env.step(action)
    #     time.sleep(1)
    # quit()

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

    # show_map(Lift_Pitch_hmap)

    Thrust_hmap = map_clipper(norm_hmap, blade_pos, y_clip=30)

    Lift_Pitch_hmap = Lift_Pitch_hmap.reshape(1, 1, Lift_Pitch_hmap.shape[0], Lift_Pitch_hmap.shape[1])
    Thrust_hmap = Thrust_hmap.reshape(1, 1, Thrust_hmap.shape[0], Thrust_hmap.shape[1])

    L_P_des = lift_pitch_model.predict(Lift_Pitch_hmap)[0]
    x_deses = x_model.predict(Thrust_hmap)[0] * 3

    lifts = L_P_des[[np.arange(0,20,2)]] * 50 + 150
    pitches = L_P_des[[np.arange(1,20,2)]] * 230 + 50
    lift_des = lifts[-1]-4
    pitch_des = pitches[-1]

    x_err = x_deses - obs['x_blade']
    min_x_err = np.where(x_err > 0.1)
    if len(min_x_err[0]) == 0:
        x_des = obs['x_blade'] + 0.03
    else:
        x_des = np.min(x_err[min_x_err]) + obs['x_blade']

    return [x_des, obs['y_blade'], lift_des, pitch_des]

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

    # des_sample_time = 0.1  # 10 Hz
    # x_model = load_model('/home/sload/Downloads/new_test_all_recordings_x_model_10_pred')
    # lift_pitch_model = load_model('/home/sload/Downloads/new_test_new_recordings_LP_model_10_pred')

    x_model = load_model('/home/sload/Downloads/lift_task_x_model_10_pred')
    lift_pitch_model = load_model('/home/sload/Downloads/lift_task_LP_model_10_pred')

    push_pid = LLC_pid.PushPid()

    X, Y, X_des, Y_des = [], [], [], []

    counter = 0
    last_loc = obs['x_vehicle']

    # timing_test()

    while True:
        obs = env.get_obs()
        des = desired_config(obs, x_model, lift_pitch_model)
        print('curr lift = ', obs['lift'], 'des lift = ', des[2])
        # print('curr pitch = ', obs['pitch'], 'des pitch = ', des[3])


    for step in range(3):

        ##### push mission #####
        while True:
            # start_time = time.time()

            des = desired_config(obs, x_model, lift_pitch_model)

            # save locations
            X.append(obs['x_blade'])
            Y.append(obs['y_blade'])
            X_des.append(des[0])
            Y_des.append(des[1])

            # action = push_pid.step(obs, des)
            # print(obs['z_vehicle'])
            # print('desired lift = ', des[2], 'current = ', obs['lift'])
            # print('desired pitch = ', des[3], 'current = ', obs['pitch'])

            action = push_pid.step(obs, des)

            while (des[2] - obs['lift']) > 8:
                print('lifting!!')
                lift_action = action
                lift_action[1] = 0
                obs = env.step(lift_action)
                des = desired_config(obs, x_model, lift_pitch_model)

            # action = np.array([0,0,0,0])
            # action[0]=0
            # action[1] = 0
            obs = env.step(action)

            movement = abs(last_loc-obs['x_vehicle'])
            print(movement)
            if movement < 0.003:
                counter += 1
                if counter == 100:
                    print('got stuck! move back')
                    break
            else:
                counter = 0
                last_loc = obs['x_vehicle']

            if obs['x_blade'] >= 2.0:
                print('got to end point!')
                break

        # plots
        push_pid.lift_pid.save_plot('lift push {}'.format(str(step)), 'lift')
        push_pid.pitch_pid.save_plot('pitch push {}'.format(str(step)), 'pitch')
        push_pid.speed_pid.save_plot('speed push {}'.format(str(step)), 'speed')

        ##### drive backwards #####
        des = [0.5, obs['y_vehicle'], 155, 125]
        back_pid = LLC_pid.DriveBackPid(des)
        while True:
            action = back_pid.step(obs, des)
            obs = env.step(action)

            # save locations
            X.append(obs['x_blade'])
            Y.append(obs['y_blade'])
            X_des.append(des[0])
            Y_des.append(des[1])

            # stopping condition
            if obs['lift'] <= des[2] and obs['pitch'] <= des[3]:
                print('driving backwards mission done!')
                break

        # plots
        back_pid.speed_pid.save_plot('speed back {}'.format(str(step)), 'speed')
        back_pid.pitch_pid.save_plot('pitch back {}'.format(str(step)), 'pitch')
        back_pid.lift_pid.save_plot('lift back {}'.format(str(step)), 'lift')

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
