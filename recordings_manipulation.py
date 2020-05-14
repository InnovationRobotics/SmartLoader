import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import ion, gca

from keras.models import Model, load_model
import re
from os import walk
import csv
import sys
import time


csv.field_size_limit(sys.maxsize)

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


def str_translator(str_val):
    if len(str_val) == 0:  # if no imu data, skip line
        global nd_skipped_data
        nd_skipped_data += 1
        return
    sns_val = []
    for cord in str_val:
        try:
            sns_val.append(float(re.sub("[^\d\.\-\'e']", "", cord)))
        except:
            continue
    return sns_val


def show_heatmap(heatmap, arm_height, anim=True):
    if anim:
        plt.imshow(heatmap, aspect=1)
        plt.text(40, 5, arm_height, fontsize=15)
        plt.show(block=False)
        plt.pause(0.01)
        try:
            del gca().texts[-1]
        except:
            pass

    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(heatmap)
        print(arm_height)
        # est_pos = [shovle_pos[0]*260, shovle_pos[1]*(y_map_clip[1]-y_map_clip[0])]
        # plt.scatter(est_pos[1], est_pos[0], s=300, c='red', marker='o')
        plt.show()

jobs = ['heat_map_generator', 'concat_recodrings', 'pos_aprox', 'pos_deductor']
job = jobs[0]
sp_skipped_data = 0
nd_skipped_data = 0
nd_skipped_data = 0
hm_skipped_data = 0

if job == 'heat_map_generator':

    recordings_path = '/home/graphics/BC_recordings/lift_pile_12_05/'

    show_map = False


    actions = []
    states = []
    heatmaps = []
    starts = []

    prev_body_pose = [0, 0, 0]

    ep_counter = 1
    step_counter = 0


    for (dirpath, dirnames, _) in walk(recordings_path):  # crate a list of episodes
        break

    for ep_dir in dirnames:  ## iterate over each
        print(ep_dir)
        starts.append(True)
        for (_, _, filenames) in walk(recordings_path+ep_dir):  # create a list of pointcloud steps for each episode
            break

        # first remove bag file from list and go over csv:
        csv_index = [n for n, x in enumerate(filenames) if 'csv' in x][0]
        bag_index = [n for n, x in enumerate(filenames) if 'bag' in x][0]

        # csv_file = filenames[csv_index][7:-1]  ## if csv is open
        csv_file = filenames[csv_index]

        with open(recordings_path+ep_dir+'/'+csv_file) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)

            for row_num, row in enumerate(reader):
                step_counter += 1

                # read actions
                act = row[2]
                if len(act) == 0:  # if no action data, skip line
                    nd_skipped_data += 1
                    continue
                act = act.replace('(', '').replace(')', '')
                act = act.split(',')
                act = [float(i) for i in act]


                # read states (i.e. arm height and pitch and body and shovle orientation)
                body_pos_str = row[3][0:86].split()
                body_pos = str_translator(body_pos_str)
                if not body_pos:
                    continue


                if prev_body_pose == body_pos:
                    # print('same pos, skipping row {}', format(row_num+1))
                    sp_skipped_data += 1
                    continue

                # if (body_pos[0]-prev_body_pose[0]) < -0.7:
                #     starts.append(True)
                # else:
                #     starts.append(False)

                prev_body_pose = body_pos

                shovle_pos_str = row[4][0:86].split()
                shovle_pos = str_translator(shovle_pos_str)
                if not shovle_pos:
                    continue

                arm_height = int(row[5])
                arm_pitch = int(row[6])

                blade_orien_str = row[7][0:120].split()
                blade_orien = str_translator(blade_orien_str)

                body_orien_str = row[8][0:120].split()
                body_orien = str_translator(body_orien_str)

                heat_map_str = row[9][178:-1].split(',')
                heat_map = str_translator(heat_map_str)

                heat_map_size = [int(row[9][64:67]),int(row[9][132:135])]

                hm_array = np.array(heat_map)
                dirt = np.where((hm_array > 0.5) | (hm_array < -0.5))
                for arr_cell in dirt[0]:
                    hm_array[arr_cell] = np.median(hm_array)

                # show_heatmap(np.array(heat_map).reshape(heat_map_size), arm_height)

                ## normalize heat_map
                heat_map = (hm_array - np.min(hm_array)) / np.ptp(hm_array)

                ## reshape heatmap
                heat_map = np.array(heat_map).reshape(heat_map_size)
                if show_map:
                    show_heatmap(heat_map, arm_height)

                # ---- append current map, state and actions to list ----
                state = [body_pos, shovle_pos, body_orien, blade_orien, arm_height, arm_pitch]
                states.append(state)
                heatmaps.append(heat_map)  # create a list of clipped heatmaps for each episode
                actions.append(joy_to_agent(act))

                starts.append(False)
                print('appended {} steps'.format(row_num))

            print('episode {}:, {} steps appended out of {}. {} skipped frames due to missing data, {} due to duplicate data and {} due to heat map limits'.format(
                ep_counter, len(states) ,step_counter, nd_skipped_data, sp_skipped_data, hm_skipped_data))
            starts.pop(-1)
            ep_counter += 1

    np.save(dirpath + 'heatmaps', heatmaps)
    np.save(dirpath + 'states', states)
    np.save(dirpath + 'starts', starts)
    np.save(dirpath + 'actions', actions)

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


if job == 'xy_labels_saver':

    recordings_path = '/home/sload/xy_locations_03_05/'

    first_ep = True
    labels = []

    for (dirpath, dirnames, _) in walk(recordings_path):  # crate a list of episodes
        break
    for ep_dir in dirnames:  ## iterate over each
        for (_, _, filenames) in walk(dirpath + ep_dir):  # create a list of pointcloud steps for each episode
            break

        popper = []

        for file_num, file_name in enumerate(filenames):
            if any(x in file_name for x in ['bag', 'csv', 'npy']):
                popper.append(file_num)
                continue

        for index in sorted(popper, reverse=True):
            del filenames[index]

        # save labels
        x_label = ep_dir.split('_')[2]
        y_label = ep_dir.split('_')[4]
        lb_size = len(filenames)
        label = np.array([lb_size * [x_label], lb_size * [y_label]])

        if first_ep:
            labels = label
            first_ep = False
        else:
            labels = np.concatenate((labels, label), axis=1)

    np.save(recordings_path + 'labels', labels)

