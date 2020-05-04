import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
from keras.models import Model, load_model
import re
from os import walk
import csv
import time


def joy_to_agent(joy_actions):
    agent_action = np.zeros(4)

    agent_action[0] = joy_actions[0]  # vehicle turn
    agent_action[2] = joy_actions[3]  # blade pitch     ##### reduced state space
    agent_action[3] = joy_actions[4]  # arm up/down
    # translate 5 dim joystick actions to 4 dim agent action
    # agent action: [steer, speed, blade_pitch, arm_height]
    # simulation joystick actions: [steer, speed backwards, blade pitch, arm height, speed forwards]
    # all [-1, 1]
    agent_action[1] = 0.5 * (joy_actions[2] - 1) + 0.5 * (1 - joy_actions[5])  ## forward backward

    return agent_action


jobs = ['heat_map_generator', 'concat_recodrings', 'pos_aprox', 'pos_deductor']
job = jobs[0]

if job == 'heat_map_generator':

    recordings_path = '/home/graphics/push_28_04_all/'

    stripe_limits = [-0.8, 0.8]

    show_point_cloud = False
    show_height_map = False

    actions = []
    states = []
    heatmaps = []
    starts = []

    ep_counter = 1
    skipped_data = 0

    stripe_range = np.arange(stripe_limits[0], stripe_limits[1] + 0.1, 0.1)
    num_of_stripes = len(stripe_range) - 1
    x_res = 100

    for (dirpath, dirnames, _) in walk(recordings_path):  # crate a list of episodes
        break
    for ep_dir in dirnames:  ## iterate over each
        for (_, _, filenames) in walk(dirpath + ep_dir):  # create a list of pointcloud steps for each episode
            break

        # first remove bag file from list and go over csv:
        csv_index = [n for n, x in enumerate(filenames) if 'csv' in x][0]
        bag_index = [n for n, x in enumerate(filenames) if 'bag' in x][0]

        csv_file = filenames[csv_index]

        for index in sorted([csv_index, bag_index], reverse=True):
            del filenames[index]

        with open(dirpath + ep_dir + '/' + csv_file) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            starts.append(True)

            for row in reader:

                # read actions
                act = row[2]
                act = act.replace('(', '').replace(')', '')
                act = act.split(',')
                act = [float(i) for i in act]

                # read states (i.e. arm heigh and pitch and body and shovle orientation)
                body_orien_str = row[8][0:107].split()
                if len(body_orien_str) == 0:  # if no imu data, skip line
                    skipped_data += 1
                    continue
                body_orien = []
                for cord in body_orien_str:
                    try:
                        body_orien.append(float(re.sub("[^\d\.\-\'e']", "", cord)))
                    except:
                        continue

                blade_orien_str = row[7][0:110].split()
                if len(blade_orien_str) == 0:  # if no imu data, skip line
                    skipped_data += 1
                    continue
                blade_orien = []
                for cord in blade_orien_str:
                    try:
                        blade_orien.append(float(re.sub("[^\d\.\-\'e']", "", cord)))
                    except:
                        continue

                state = [int(row[5]), int(row[6]), body_orien, blade_orien]

                # read heatmap
                h_map_file = row[10] + '.npz'

                point_cloud = np.load(dirpath + ep_dir + '/' + h_map_file)['arr_0']
                if len(point_cloud) < 10000:
                    skipped_data += 1
                    continue

                start_time = time.time()

                z = -point_cloud[:, 0]
                x = point_cloud[:, 1]
                y = -point_cloud[:, 2]

                ind = np.where((x < -1.4) | (x > 1.5) | (z > -2.2))

                x = np.delete(x, ind)
                y = np.delete(y, ind)
                z = np.delete(z, ind)

                z = (z - np.min(z)) / np.ptp(z)  ## normalize height [0,1]

                if show_point_cloud:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    ax.scatter(x, y, z, s=0.1)

                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')
                    plt.show()

                h_map = np.zeros([x_res, num_of_stripes])

                x_A_coeff = (x_res - 1) / 3
                x_B_coeff = (x_res - 1) / 2

                y_A_coeff = (num_of_stripes - 1) / (stripe_limits[1] - stripe_limits[0])
                y_B_coeff = -stripe_limits[0] * y_A_coeff

                x_ind = np.array([x * x_A_coeff + x_B_coeff]).astype(int)
                y_ind = np.array([y * y_A_coeff + y_B_coeff]).astype(int)

                for y_step in range(num_of_stripes):

                    for x_step in range(x_res):

                        z_ind = np.argwhere(np.logical_and(y_step == y_ind, x_step == x_ind)[0])
                        if len(z_ind) == 0:
                            continue
                        h_map[x_step, y_step] = np.max(z[z_ind])

                # pc_len = len(x)
                #
                # for point in range(pc_len):
                #     if (y[point] < stripe_limits[0]) | (y[point] > stripe_limits[1]+0.1):
                #         continue
                #     x_ind = int(x[point] * x_A_coeff + x_B_coeff)
                #     y_ind = int(y[point] * y_A_coeff + y_B_coeff)
                #     if z[point] > h_map[x_ind, y_ind]:
                #         h_map[x_ind, y_ind] = z[point]

                frame_time = time.time() - start_time
                print(frame_time)

                if show_height_map:
                    plt.imshow(h_map, aspect=0.1)
                    plt.show(block=False)
                    plt.pause(0.001)

                heatmaps.append(h_map)  # create a list of heatmaps for each episode
                starts.append(False)
                actions.append(joy_to_agent(act))
                states.append(state)

            starts.pop(-1)
            print('episode {} appended, {} skipped frames due to missing data or low quality pointclouds'.format(
                ep_counter, skipped_data))
            skipped_data = 0
            ep_counter += 1

    np.save(recordings_path + 'heatmaps', heatmaps)
    np.save(recordings_path + 'states', states)
    np.save(recordings_path + 'starts', starts)
    np.save(recordings_path + 'actions', actions)

if job == 'concat_recodrings':
    act1 = np.load('/home/graphics/git/SmartLoader/saved_experts/saved_ep/ob.npy')
    obs1 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/80_ep_5_hist/obs.npy')
    rew1 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/80_ep_5_hist/rew.npy')
    ep_ret1 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/80_ep_5_hist/ep_ret.npy')
    ep_str1 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/80_ep_5_hist/ep_str.npy')

    act2 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/20_ep_5_hist_diff/act.npy')
    obs2 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/20_ep_5_hist_diff/obs.npy')
    rew2 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/20_ep_5_hist_diff/rew.npy')
    ep_ret2 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/20_ep_5_hist_diff/ep_ret.npy')
    ep_str2 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/20_ep_5_hist_diff/ep_str.npy')

    act = np.concatenate((act1, act2))
    obs = np.concatenate((obs1, obs2))
    rew = np.concatenate((rew1, rew2))
    ep_ret = np.concatenate((ep_ret1, ep_ret2))
    ep_str = np.concatenate((ep_str1, ep_str2))

    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/act', act)
    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/obs', obs)
    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/rew', rew)
    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/ep_ret', ep_ret)
    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/ep_str', ep_str)

if job == 'pos_aprox':

    expert_path = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/Push_49_ep/full_map/'

    heat_maps = np.load(expert_path + 'heatmap.npy')

    num_of_labels = 50

    global pos

    inputs = []
    labels = []
    pos = []

    for k in range(num_of_labels):
        def onclick(event):
            global pos
            xd, yd = event.xdata, event.ydata
            print(xd, yd)
            pos.append([int(xd), int(yd)])


        fig, ax = plt.subplots(figsize=(10, 10))
        fig.canvas.mpl_connect('button_press_event', onclick)

        index = np.random.randint(len(heat_maps))
        h_map = heat_maps[index, :, :].reshape(100, 16)
        # ax.figure(figsize=(10, 10))

        ax.imshow(h_map)
        plt.show()

        inputs.append(h_map)
        print(pos)


        # for k in range(0, len(labels), 2): # for 2 coordinates - body and shovle locations
        #     t_labels.append([labels[k],labels[k+1]])

        labels.append(pos[k])  # for 1 coordinatge - body location

    np.save(expert_path + 'pos_approx_map', inputs)
    np.save(expert_path + 'pos_approx_label', labels)

if job == 'pos_deductor':

    expert_path = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/Push_49_ep/full_map/'
    heat_maps = np.load(expert_path + 'heatmap.npy')
    states = np.load(expert_path + 'states.npy', allow_pickle=True)
    starts = np.load(expert_path + 'starts.npy')
    model = load_model(expert_path+'KERAS_pos_est_model_08_val_loss')

    positions = []
    for step, h_map in enumerate(heat_maps):
        positions.append(model.predict(h_map.reshape(1, 1, 100, 16)))

    np.save(expert_path + 'positions', positions)





